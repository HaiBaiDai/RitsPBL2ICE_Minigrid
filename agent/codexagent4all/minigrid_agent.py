#!/usr/bin/env python3
"""Observation-driven baseline agent for the registered MiniGrid environments."""

from __future__ import annotations

import argparse
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments with Gymnasium
import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX
from minigrid.core.actions import Actions


IDX_TO_OBJECT = {value: key for key, value in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {value: key for key, value in COLOR_TO_IDX.items()}
COLORS = tuple(COLOR_TO_IDX)
MOVABLE = {"key", "ball", "box"}
FLOOR = {"empty", "floor", "goal"}
ACTION_NAMES = {int(action): action.name for action in Actions}


@dataclass(frozen=True)
class Cell:
    kind: str
    color: str | None = None
    state: int = 0

    @property
    def walkable(self) -> bool:
        return self.kind in FLOOR or (self.kind == "door" and self.state == 0)

    @property
    def door_open(self) -> bool:
        return self.kind == "door" and self.state == 0

    @property
    def door_locked(self) -> bool:
        return self.kind == "door" and self.state == 2


class MiniGridAgent:
    """A small map-building planner that uses only MiniGrid observations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.world: dict[tuple[int, int], Cell] = {}
        self.pos = (0, 0)
        self.direction = 0
        self.start_direction = 0
        self.carrying: Cell | None = None
        self.mission = ""
        self.task = "goal"
        self.target_color: str | None = None
        self.target_kind: str | None = None
        self.secondary_color: str | None = None
        self.secondary_kind: str | None = None
        self.memory_kind: str | None = None
        self.start_pos = (0, 0)
        self.prev_action: int | None = None
        self.prev_pos = (0, 0)
        self.prev_direction = 0
        self.steps = 0

    def _parse_mission(self, mission: str) -> None:
        if mission == self.mission:
            return
        self.mission = mission
        lower = mission.lower()
        self.target_color = None
        self.target_kind = None
        self.secondary_color = None
        self.secondary_kind = None

        putnear = re.search(
            r"put the (red|green|blue|purple|yellow|grey) (key|ball|box) near the "
            r"(red|green|blue|purple|yellow|grey) (key|ball|box)", lower,
        )
        if putnear:
            self.task = "putnear"
            self.target_color, self.target_kind = putnear.group(1), putnear.group(2)
            self.secondary_color, self.secondary_kind = putnear.group(3), putnear.group(4)
            return

        if "matching object" in lower:
            self.task = "memory"
            return

        door_target = re.search(r"go to the (red|green|blue|purple|yellow|grey) door", lower)
        if door_target:
            self.task = "goto_door"
            self.target_color, self.target_kind = door_target.group(1), "door"
            return

        object_target = re.search(
            r"go to the (red|green|blue|purple|yellow|grey) (key|ball|box)", lower
        )
        if object_target:
            self.task = "goto_object"
            self.target_color, self.target_kind = object_target.group(1), object_target.group(2)
            return

        if "open the red door" in lower and "blue door" in lower:
            self.task = "redblue"
            return

        if lower.strip() == "open the door" or lower.startswith("open the "):
            self.task = "open"
            return

        if "unlock" in lower or "locked room" in lower:
            self.task = "goal_with_key" if "key" in lower else "goal"
            key_target = re.search(r"get the (red|green|blue|purple|yellow|grey) key", lower)
            if key_target:
                self.target_color, self.target_kind = key_target.group(1), "key"
            return

        pickup = re.search(
            r"(?:pick up|fetch|get|go get|go fetch) (?:a |the )?"
            r"(red|green|blue|purple|yellow|grey) (key|ball|box)", lower
        )
        if pickup:
            self.task = "pickup"
            self.target_color, self.target_kind = pickup.group(1), pickup.group(2)
            return

        self.task = "goal"

    def _advance_pose(self, obs: dict) -> None:
        if self.prev_action is None:
            return
        action = self.prev_action
        if action == int(Actions.left):
            self.direction = (self.direction - 1) % 4
        elif action == int(Actions.right):
            self.direction = (self.direction + 1) % 4
        elif action == int(Actions.forward):
            dx, dy = self._forward_vector(self.direction)
            old_pos = self.prev_pos
            moved_pos = (old_pos[0] + dx, old_pos[1] + dy)
            # A forward step can bump into a moving obstacle or an unexplored wall.
            # Align the new view against the old map to tell whether it moved.
            old_score = self._view_mismatch(obs, old_pos, self.direction)
            moved_score = self._view_mismatch(obs, moved_pos, self.direction)
            self.pos = moved_pos if moved_score <= old_score else old_pos
        self.prev_action = None

    def _view_mismatch(self, obs: dict, candidate_pos: tuple[int, int], direction: int) -> int:
        image = obs["image"]
        size = image.shape[0]
        center, bottom = size // 2, size - 1
        mismatches = 0
        for local_x in range(size):
            for local_y in range(size):
                if local_x == center and local_y == bottom:
                    # The center cell may encode the carried object instead of floor.
                    continue
                obj_id, color_id, state = (int(v) for v in image[local_x, local_y])
                kind = IDX_TO_OBJECT.get(obj_id, "unseen")
                if kind in {"unseen", "agent"}:
                    continue
                coord = self._relative_to_world(local_x, local_y, center, bottom, candidate_pos, direction)
                known = self.world.get(coord)
                if known is None:
                    continue
                color = IDX_TO_COLOR.get(color_id)
                if known.kind != kind or (known.kind in {"door", "key", "ball", "box"} and known.color != color):
                    mismatches += 1
                elif known.kind == "door" and known.state != state:
                    mismatches += 1
        return mismatches

    @staticmethod
    def _forward_vector(direction: int) -> tuple[int, int]:
        return ((1, 0), (0, 1), (-1, 0), (0, -1))[direction]

    @staticmethod
    def _neighbors(pos: tuple[int, int]) -> Iterable[tuple[int, int]]:
        x, y = pos
        yield (x + 1, y)
        yield (x, y + 1)
        yield (x - 1, y)
        yield (x, y - 1)

    def _relative_to_world(
        self,
        local_x: int,
        local_y: int,
        center: int,
        bottom: int,
        origin: tuple[int, int] | None = None,
        direction: int | None = None,
    ) -> tuple[int, int]:
        side = local_x - center
        ahead = bottom - local_y
        origin = self.pos if origin is None else origin
        direction = self.direction if direction is None else direction
        # The observation is rotated so the agent faces up.
        if direction == 0:  # east
            dx, dy = ahead, side
        elif direction == 1:  # south
            dx, dy = -side, ahead
        elif direction == 2:  # west
            dx, dy = -ahead, -side
        else:  # north
            dx, dy = side, -ahead
        return origin[0] + dx, origin[1] + dy

    def _observe(self, obs: dict) -> None:
        if self.steps == 0:
            self.direction = int(obs.get("direction", 0))
            self.start_direction = self.direction
        self._advance_pose(obs)
        image = obs["image"]
        size = image.shape[0]
        center, bottom = size // 2, size - 1
        if self.steps == 0:
            self.start_pos = self.pos
        for local_x in range(size):
            for local_y in range(size):
                if local_x == center and local_y == bottom:
                    continue
                obj_id, color_id, state = (int(v) for v in image[local_x, local_y])
                kind = IDX_TO_OBJECT.get(obj_id, "unseen")
                if kind == "unseen":
                    continue
                color = IDX_TO_COLOR.get(color_id)
                coord = self._relative_to_world(local_x, local_y, center, bottom)
                if kind == "agent":
                    kind = "empty"
                self.world[coord] = Cell(kind, color, state)
                if self.task == "memory" and self.memory_kind is None and kind in {"key", "ball"}:
                    # Memory levels place the clue at the west end and the matching
                    # choices at the far east end; the agent starts facing east.
                    if color == "green" and coord[0] <= self.start_pos[0]:
                        self.memory_kind = kind
        self.world[self.pos] = Cell("empty")
        if "mission" in obs:
            self._parse_mission(obs["mission"])
            # A newly parsed Memory mission can occur after the first map frame.
            if self.task == "memory" and self.memory_kind is None:
                for coord, cell in self.world.items():
                    if cell.color == "green" and cell.kind in {"key", "ball"} and coord[0] <= self.start_pos[0]:
                        self.memory_kind = cell.kind
                        break
        self.steps += 1

    def _walkable(self, coord: tuple[int, int]) -> bool:
        cell = self.world.get(coord)
        return cell is not None and cell.walkable and cell.kind != "lava"

    def _route(self, goals: set[tuple[int, int]]) -> list[int] | None:
        """Shortest known safe path to any walkable goal cell."""
        if self.pos in goals:
            return []
        start = (self.pos[0], self.pos[1], self.direction)
        queue = deque([start])
        parent: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {start: None}
        finish = None
        while queue:
            state = queue.popleft()
            x, y, direction = state
            if (x, y) in goals:
                finish = state
                break
            candidates = [
                ((x, y, (direction - 1) % 4), int(Actions.left)),
                ((x, y, (direction + 1) % 4), int(Actions.right)),
            ]
            dx, dy = self._forward_vector(direction)
            ahead = (x + dx, y + dy)
            if self._walkable(ahead):
                candidates.append(((ahead[0], ahead[1], direction), int(Actions.forward)))
            for nxt, action in candidates:
                if nxt not in parent:
                    parent[nxt] = (state, action)
                    queue.append(nxt)
        if finish is None:
            return None
        actions: list[int] = []
        cursor = finish
        while parent[cursor] is not None:
            prev, action = parent[cursor]
            actions.append(action)
            cursor = prev
        actions.reverse()
        return actions

    def _facing_route(self, target: tuple[int, int]) -> list[int] | None:
        # Choose the adjacent pose with minimum path + turn cost.
        best: list[int] | None = None
        for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
            stand = (target[0] - dx, target[1] - dy)
            if not self._walkable(stand):
                continue
            route = self._route({stand})
            if route is None:
                continue
            final_dir = self.direction if not route else self._end_direction(route)
            candidate = route + self._turns_to(direction, final_dir)
            if best is None or len(candidate) < len(best):
                best = candidate
        return best

    def _end_direction(self, actions: list[int]) -> int:
        direction = self.direction
        for action in actions:
            if action == int(Actions.left):
                direction = (direction - 1) % 4
            elif action == int(Actions.right):
                direction = (direction + 1) % 4
        return direction

    @staticmethod
    def _turns_to(target_dir: int, current_dir: int) -> list[int]:
        delta = (target_dir - current_dir) % 4
        if delta == 0:
            return []
        if delta == 1:
            return [int(Actions.right)]
        if delta == 2:
            return [int(Actions.right), int(Actions.right)]
        return [int(Actions.left)]

    def _cells(self, kind: str | None = None, color: str | None = None) -> list[tuple[int, int]]:
        return [
            coord for coord, cell in self.world.items()
            if (kind is None or cell.kind == kind) and (color is None or cell.color == color)
        ]

    def _best_facing_route(self, targets: list[tuple[int, int]]) -> tuple[tuple[int, int], list[int]] | None:
        best = None
        for target in targets:
            route = self._facing_route(target)
            if route is not None and (best is None or len(route) < len(best[1])):
                best = (target, route)
        return best

    def _front_coord(self) -> tuple[int, int]:
        dx, dy = self._forward_vector(self.direction)
        return self.pos[0] + dx, self.pos[1] + dy

    def _front_cell(self) -> Cell | None:
        return self.world.get(self._front_coord())

    def _issue(self, action: int) -> int:
        self.prev_action = action
        self.prev_pos = self.pos
        self.prev_direction = self.direction
        front = self._front_coord()
        if action == int(Actions.pickup):
            item = self.world.get(front)
            if item and item.kind in MOVABLE:
                self.carrying = item
                self.world[front] = Cell("empty")
        elif action == int(Actions.drop) and self.carrying:
            if self.world.get(front, Cell("wall")).kind in FLOOR:
                self.world[front] = self.carrying
                self.carrying = None
        elif action == int(Actions.toggle):
            door = self.world.get(front)
            if door and door.kind == "door":
                self.world[front] = Cell("door", door.color, 0 if door.state != 0 else 1)
            elif door and door.kind == "box":
                # MiniGrid boxes reveal their contents when toggled.
                self.world[front] = Cell("empty")
        return action

    def _do_route(self, route: list[int] | None) -> int | None:
        if route is None:
            return None
        return self._issue(route[0]) if route else None

    def _target_objects(self) -> list[tuple[int, int]]:
        return self._cells(self.target_kind, self.target_color)

    def _frontier_action(self) -> int | None:
        candidates: list[tuple[int, list[int]]] = []
        for stand, cell in self.world.items():
            if not cell.walkable:
                continue
            route = self._route({stand})
            if route is None:
                continue
            final_dir = self._end_direction(route) if route else self.direction
            for unknown in self._neighbors(stand):
                if unknown in self.world:
                    continue
                dx, dy = unknown[0] - stand[0], unknown[1] - stand[1]
                target_dir = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(dx, dy)]
                plan = route + self._turns_to(target_dir, final_dir) + [int(Actions.forward)]
                candidates.append((len(plan), plan))
        if candidates:
            return self._issue(min(candidates, key=lambda item: item[0])[1][0])
        return None

    def _keys_for_locked_doors(self) -> list[tuple[int, int]]:
        colors_needed = {cell.color for cell in self.world.values() if cell.door_locked}
        if self.task == "pickup" and self.target_kind == "key":
            colors_needed.discard(self.target_color)
        if self.task == "putnear" and self.target_kind == "key":
            colors_needed.discard(self.target_color)
        return [coord for coord, cell in self.world.items() if cell.kind == "key" and cell.color in colors_needed]

    def _closed_doors(self) -> list[tuple[int, int]]:
        return [coord for coord, cell in self.world.items() if cell.kind == "door" and cell.state != 0]

    def _handle_doors(self) -> int | None:
        if self.task in {"goto_door", "goto_object"}:
            return None
        if self.task == "redblue":
            # RedBlueDoors requires opening red before blue.
            red_open = any(cell.kind == "door" and cell.color == "red" and cell.door_open for cell in self.world.values())
            color = "blue" if red_open else "red"
            doors = [p for p in self._cells("door", color) if self.world[p].state != 0]
            if not doors:
                return None
            selected = self._best_facing_route(doors)
            if not selected:
                return None
            target, route = selected
            if route:
                return self._issue(route[0])
            return self._issue(int(Actions.toggle))

        doors = self._closed_doors()
        if not doors:
            return None
        for door_pos in doors:
            action = self._handle_blocking_ball(door_pos)
            if action is not None:
                return action
        if self.task == "pickup" and self.target_kind == "ball" and self.carrying and self.carrying.kind == "key":
            if not any(self.world[p].door_locked and self.world[p].color == self.carrying.color for p in doors):
                action = self._drop_hand()
                if action is not None:
                    return action
        if self.carrying is None:
            keys = self._keys_for_locked_doors()
            selected_key = self._best_facing_route(keys)
            if selected_key:
                _, route = selected_key
                if route:
                    return self._issue(route[0])
                return self._issue(int(Actions.pickup))
            if self.task == "pickup" and self.target_kind == "ball":
                boxes = self._cells("box")
                selected_box = self._best_facing_route(boxes)
                if selected_box:
                    _, route = selected_box
                    return self._issue(route[0]) if route else self._issue(int(Actions.toggle))
        # Open reachable unlocked doors, and only unlock a door with a matching key.
        available_colors = {self.carrying.color} if self.carrying and self.carrying.kind == "key" else set()
        available_colors.update(self.world[p].color for p in self._keys_for_locked_doors())
        candidates = []
        for door_pos in doors:
            door = self.world[door_pos]
            if door.door_locked and door.color not in available_colors:
                continue
            selected = self._best_facing_route([door_pos])
            if selected:
                candidates.append((len(selected[1]), door_pos, selected[1]))
        if candidates:
            _, door_pos, route = min(candidates, key=lambda item: item[0])
            if route:
                return self._issue(route[0])
            door = self.world[door_pos]
            if door.door_locked and (not self.carrying or self.carrying.kind != "key" or self.carrying.color != door.color):
                return None
            return self._issue(int(Actions.toggle))

        return None

    def _handle_blocking_ball(self, door_pos: tuple[int, int]) -> int | None:
        move_ball = self.task == "pickup" and (
            self.target_kind == "box" or self.target_kind == "ball"
        )
        target_color = self.target_color if self.target_kind == "ball" else None
        balls = [
            p for p in self._neighbors(door_pos)
            if (cell := self.world.get(p)) and cell.kind == "ball"
            and (target_color is None or cell.color != target_color)
        ]
        carrying_blocker = (
            self.carrying and self.carrying.kind == "ball"
            and (self.target_kind == "box" or (self.target_kind == "ball" and self.carrying.color != self.target_color))
        )
        if carrying_blocker and move_ball:
            # BlockedUnlockPickup requires moving the ball before returning for the key.
            door_positions = self._closed_doors() or [door_pos]
            options = []
            for stand, cell in self.world.items():
                if not cell.walkable or cell.kind == "goal":
                    continue
                route = self._route({stand})
                if route is None:
                    continue
                final_dir = self._end_direction(route) if route else self.direction
                for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                    landing = (stand[0] + dx, stand[1] + dy)
                    landing_cell = self.world.get(landing)
                    if landing_cell is None or landing_cell.kind not in FLOOR or landing_cell.kind == "goal":
                        continue
                    distances = [abs(landing[0] - door[0]) + abs(landing[1] - door[1]) for door in door_positions]
                    if any(distance <= 1 for distance in distances):
                        continue
                    plan = route + self._turns_to(direction, final_dir)
                    distance = min(distances)
                    options.append(((-distance, len(plan)), plan))
            if options:
                plan = min(options, key=lambda item: item[0])[1]
                return self._issue(plan[0]) if plan else self._issue(int(Actions.drop))
            return None
        if not balls:
            return None
        if self.carrying and self.carrying.kind != "ball":
            return self._drop_hand()
        if self.carrying and self.carrying.kind == "ball":
            # Move the blocker well away from the doorway so it does not close the route again.
            door_positions = self._closed_doors() or [door_pos]
            options = []
            for stand, cell in self.world.items():
                if not cell.walkable or cell.kind == "goal":
                    continue
                route = self._route({stand})
                if route is None:
                    continue
                final_dir = self._end_direction(route) if route else self.direction
                for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                    landing = (stand[0] + dx, stand[1] + dy)
                    landing_cell = self.world.get(landing)
                    if landing_cell is None or landing_cell.kind not in FLOOR or landing_cell.kind == "goal":
                        continue
                    distances = [abs(landing[0] - door[0]) + abs(landing[1] - door[1]) for door in door_positions]
                    if any(distance <= 1 for distance in distances):
                        continue
                    plan = route + self._turns_to(direction, final_dir)
                    distance = min(distances)
                    options.append(((-distance, len(plan)), plan))
            if options:
                plan = min(options, key=lambda item: item[0])[1]
                return self._issue(plan[0]) if plan else self._issue(int(Actions.drop))
            return None
        selected = self._best_facing_route(balls)
        if selected:
            _, route = selected
            return self._issue(route[0]) if route else self._issue(int(Actions.pickup))
        return None

    def _drop_hand(self) -> int | None:
        if self.carrying is None:
            return None
        possible = []
        for stand, cell in self.world.items():
            if not cell.walkable or cell.kind == "goal":
                continue
            for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                front_pos = (stand[0] + dx, stand[1] + dy)
                front_cell = self.world.get(front_pos)
                if front_cell is not None and front_cell.kind in FLOOR:
                    route = self._route({stand})
                    if route is not None:
                        candidate = route + self._turns_to(direction, self._end_direction(route) if route else self.direction)
                        distance_to_start = abs(front_pos[0] - self.start_pos[0]) + abs(front_pos[1] - self.start_pos[1])
                        possible.append(((distance_to_start, len(candidate)), candidate))
        if possible:
            route = min(possible, key=lambda item: item[0])[1]
            if route:
                return self._issue(route[0])
            return self._issue(int(Actions.drop))
        return None

    def _memory_target(self) -> list[tuple[int, int]]:
        if self.memory_kind is None:
            return []
        candidates = [
            coord for coord, cell in self.world.items()
            if cell.kind == self.memory_kind and cell.color == "green"
            and (abs(coord[0] - self.start_pos[0]) > 1 or abs(coord[1] - self.start_pos[1]) > 1)
        ]
        if not candidates:
            return []
        farthest = max(coord[0] for coord in candidates)
        objects = [coord for coord in candidates if coord[0] == farthest]
        # The success square is the inner hallway tile beside the matching object.
        return [(x, y + 1 if y < self.start_pos[1] else y - 1) for x, y in objects]

    def _putnear_action(self) -> int | None:
        landmarks = self._cells(self.secondary_kind, self.secondary_color)
        if not landmarks:
            return None
        targets = self._target_objects()
        if self.carrying is None:
            selected = self._best_facing_route(targets)
            if selected:
                _, route = selected
                if route:
                    return self._issue(route[0])
                return self._issue(int(Actions.pickup))
            return None
        drop_cells = set()
        for landmark in landmarks:
            for coord in self._neighbors(landmark):
                cell = self.world.get(coord)
                if cell and cell.kind in FLOOR and cell.kind != "goal":
                    drop_cells.add(coord)
        candidates = []
        for drop_pos in drop_cells:
            for direction, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1))):
                stand = (drop_pos[0] - dx, drop_pos[1] - dy)
                if self._walkable(stand):
                    route = self._route({stand})
                    if route is not None:
                        final_dir = self._end_direction(route) if route else self.direction
                        candidate = route + self._turns_to(direction, final_dir)
                        candidates.append((len(candidate), candidate))
        if candidates:
            route = min(candidates, key=lambda item: item[0])[1]
            if route:
                return self._issue(route[0])
            return self._issue(int(Actions.drop))
        return None

    def act(self, obs: dict) -> int:
        self._observe(obs)
        # Tasks that are solved by standing beside an object/door use Actions.done.
        if self.task == "goto_door":
            targets = self._cells("door", self.target_color)
            selected = self._best_facing_route(targets)
            if selected:
                _, route = selected
                return self._issue(route[0]) if route else self._issue(int(Actions.done))
        elif self.task == "goto_object":
            targets = self._target_objects()
            selected = self._best_facing_route(targets)
            if selected:
                _, route = selected
                return self._issue(route[0]) if route else self._issue(int(Actions.done))
        elif self.task == "memory":
            targets = self._memory_target()
            route = self._route(set(targets)) if targets else None
            if route is not None:
                return self._issue(route[0]) if route else self._issue(int(Actions.done))
        elif self.task == "putnear":
            action = self._putnear_action()
            if action is not None:
                return action
        elif self.task in {"pickup", "goal_with_key"} and self.target_kind:
            targets = self._target_objects()
            # A held target object completes a fetch/pickup mission immediately.
            if self.task == "pickup" and self.carrying and self.carrying.kind == self.target_kind and self.carrying.color == self.target_color:
                return self._issue(int(Actions.done))
            locked_needs_key = any(c.door_locked for c in self.world.values())
            if self.task == "pickup" and targets:
                # The target is already reachable; free the hand so it can be picked up.
                locked_needs_key = False
            if self.carrying and not locked_needs_key and not (
                self.task == "goal_with_key" and self.carrying.kind == "key"
            ):
                action = self._drop_hand()
                if action is not None:
                    return action
            selected = self._best_facing_route(targets)
            if selected:
                _, route = selected
                if route:
                    return self._issue(route[0])
                if self.carrying is None:
                    return self._issue(int(Actions.pickup))
        if self.task == "goal" or self.task == "goal_with_key":
            goals = self._cells("goal")
            route = self._route(set(goals)) if goals else None
            if route is not None:
                if route:
                    return self._issue(route[0])
                return self._issue(int(Actions.done))

        # Unlock one useful closed door at a time, collecting its matching key first.
        action = self._handle_doors()
        if action is not None:
            return action

        # If the mission target is known but still blocked, exploration or door handling
        # will reveal the route; otherwise continue to the nearest unexplored boundary.
        action = self._frontier_action()
        if action is not None:
            return action

        # Fallback: try any reachable locked door if its key has been acquired.
        for door_pos in self._closed_doors():
            door = self.world[door_pos]
            if door.door_locked and self.carrying and self.carrying.kind == "key" and self.carrying.color == door.color:
                selected = self._best_facing_route([door_pos])
                if selected:
                    _, route = selected
                    return self._issue(route[0]) if route else self._issue(int(Actions.toggle))
        return self._issue(int(Actions.done))


def run_episode(env_id: str, seed: int, render: bool, delay: float, max_steps: int) -> tuple[bool, int, float, str]:
    env = gym.make(env_id, render_mode="human" if render else None)
    try:
        obs, _ = env.reset(seed=seed)
        agent = MiniGridAgent()
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if render and delay > 0:
                time.sleep(delay)
            if terminated or truncated:
                return total_reward > 0, step, total_reward, "terminated" if terminated else "truncated"
        return False, max_steps, total_reward, "step_limit"
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="Gymnasium environment ID")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true", help="open MiniGrid's human-rendered window")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds to pause after each rendered action")
    args = parser.parse_args()

    successes = 0
    for episode in range(args.episodes):
        try:
            success, steps, reward, end_reason = run_episode(
                args.env, args.seed + episode, args.render, args.delay, args.max_steps
            )
            successes += int(success)
            print(
                f"env={args.env} episode={episode + 1}/{args.episodes} "
                f"success={int(success)} steps={steps} reward={reward:.4f} end={end_reason}",
                flush=True,
            )
        except Exception as exc:
            print(f"env={args.env} episode={episode + 1}/{args.episodes} error={type(exc).__name__}: {exc}", flush=True)
            raise
    print(f"summary env={args.env} successes={successes}/{args.episodes}")


if __name__ == "__main__":
    main()
