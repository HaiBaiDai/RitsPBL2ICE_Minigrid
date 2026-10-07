"""Observation-based baseline agent for common MiniGrid tasks.

The agent keeps a small map from what it has observed, then uses breadth-first
search over position, direction, key possession, and opened doors. It does not
read the environment's hidden grid.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import re
import time
from typing import Callable, Iterable

import gymnasium as gym
import minigrid  # noqa: F401  Ensures MiniGrid environments are registered.
import numpy as np
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid


# MiniGrid's direction numbering is east, south, west, north.
FORWARD = ((1, 0), (0, 1), (-1, 0), (0, -1))


@dataclass(frozen=True)
class Cell:
    kind: str
    color: str | None = None
    is_open: bool = False
    is_locked: bool = False
    can_overlap: bool = True
    can_pickup: bool = False


@dataclass(frozen=True)
class SearchState:
    x: int
    y: int
    direction: int
    has_key: bool
    opened_doors: frozenset[tuple[int, int]]

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)


class MiniGridAgent:
    """Observation-based planner for five common MiniGrid task families."""

    def __init__(self, environment_id: str | None = None) -> None:
        self.environment_id = (environment_id or "").lower()
        self.reset()

    def reset(self) -> None:
        self.pos = (0, 0)
        self.direction = 0
        self.has_key = False
        self.carrying_kind: str | None = None
        self.carrying_color: str | None = None
        self.world: dict[tuple[int, int], Cell] = {}
        self.mission = ""
        self.requires_key = False
        self.target_kind: str | None = None
        self.target_color: str | None = None
        self.dynamic_mode = "dynamic-obstacles" in self.environment_id
        self.red_blue_mode = "redbluedoors" in self.environment_id
        self.red_opened = False
        self.pending: tuple[int, tuple[int, int], int, tuple[int, int], Cell | None] | None = None

    @staticmethod
    def _front(pos: tuple[int, int], direction: int) -> tuple[int, int]:
        dx, dy = FORWARD[direction]
        return (pos[0] + dx, pos[1] + dy)

    @staticmethod
    def _cell_from_obj(obj: object | None) -> Cell:
        if obj is None:
            return Cell("empty")
        overlap = getattr(obj, "can_overlap", False)
        if callable(overlap):
            overlap = overlap()
        pickup = getattr(obj, "can_pickup", False)
        if callable(pickup):
            pickup = pickup()
        return Cell(
            kind=getattr(obj, "type", "unknown"),
            color=getattr(obj, "color", None),
            is_open=bool(getattr(obj, "is_open", False)),
            is_locked=bool(getattr(obj, "is_locked", False)),
            can_overlap=bool(overlap),
            can_pickup=bool(pickup),
        )

    def _apply_previous_action(self) -> list[tuple[int, int]] | None:
        if self.pending is None:
            return None

        action, old_pos, old_direction, front_pos, front_cell = self.pending
        position_candidates = None
        if action == int(Actions.forward):
            # Unknown tiles are entered only while exploring a visible frontier.
            if front_cell is None or self._walkable(front_cell):
                if self.dynamic_mode and (front_cell is None or front_cell.kind != "goal"):
                    # A moving obstacle can enter the cell between observation
                    # and the environment transition, blocking the move.
                    # Resolve that uncertainty from static landmarks in the next view.
                    position_candidates = [front_pos, old_pos]
                    self.pos = old_pos
                else:
                    self.pos = front_pos
            else:
                self.pos = old_pos
        elif action == int(Actions.pickup) and front_cell is not None and front_cell.can_pickup:
            if front_cell.kind == "key":
                self.has_key = True
            self.carrying_kind = front_cell.kind
            self.carrying_color = front_cell.color
            self.world[front_pos] = Cell("empty")
        elif action == int(Actions.drop) and self.carrying_kind is not None:
            if front_cell is None or front_cell.kind == "empty":
                self.world[front_pos] = Cell(
                    self.carrying_kind,
                    color=self.carrying_color,
                    can_overlap=self.carrying_kind == "key",
                    can_pickup=True,
                )
                self.carrying_kind = None
                self.carrying_color = None
        elif action == int(Actions.toggle) and front_cell is not None:
            if front_cell.kind == "door" and not front_cell.is_open:
                if not front_cell.is_locked or self.has_key:
                    self.world[front_pos] = Cell(
                        "door", front_cell.color, is_open=True, is_locked=False, can_overlap=True
                    )
                    if front_cell.color == "red":
                        self.red_opened = True

        self.direction = old_direction
        self.pending = None
        return position_candidates

    @staticmethod
    def _walkable(
        cell: Cell,
        opened_doors: frozenset[tuple[int, int]] = frozenset(),
        pos: tuple[int, int] | None = None,
    ) -> bool:
        if cell.kind == "lava":
            return False
        if cell.kind == "door":
            return cell.is_open or (pos is not None and pos in opened_doors)
        return cell.can_overlap

    def _observe(
        self,
        observation: dict,
        position_candidates: list[tuple[int, int]] | None = None,
    ) -> None:
        """Merge the visible egocentric image into the agent's relative map."""
        image = np.asarray(observation["image"])
        view, visible = Grid.decode(image)
        width, height = image.shape[:2]
        center_x = width // 2
        agent_view_y = height - 1

        self.direction = int(observation["direction"])
        if position_candidates and len(position_candidates) > 1:
            self.pos = self._best_position(view, visible, position_candidates, self.direction)
        forward_x, forward_y = FORWARD[self.direction]
        right_x, right_y = -forward_y, forward_x

        # The agent's cell is overwritten in the image by its carried object.
        # Preserve the underlying map cell there; use the image only to detect
        # that a key is being carried.
        current_obj = view.get(center_x, agent_view_y)
        carried_type = getattr(current_obj, "type", None)
        if carried_type in {"key", "box", "ball"}:
            self.carrying_kind = carried_type
            self.carrying_color = getattr(current_obj, "color", None)
            if carried_type == "key":
                self.has_key = True
        if self.pos not in self.world:
            self.world[self.pos] = Cell("empty")

        for view_x in range(width):
            for view_y in range(height):
                if not visible[view_x, view_y]:
                    continue
                if view_x == center_x and view_y == agent_view_y:
                    continue

                lateral = view_x - center_x
                ahead = agent_view_y - view_y
                world_pos = (
                    self.pos[0] + forward_x * ahead + right_x * lateral,
                    self.pos[1] + forward_y * ahead + right_y * lateral,
                )
                self.world[world_pos] = self._cell_from_obj(view.get(view_x, view_y))

        self.mission = str(observation.get("mission", ""))
        mission_lower = self.mission.lower()
        target = re.search(r"\bpick up the\s+(\w+)\s+(\w+)\b", mission_lower)
        if target:
            self.target_color, self.target_kind = target.groups()
        if "red" in mission_lower and "blue" in mission_lower:
            self.red_blue_mode = True
        if any(
            cell.kind == "door" and cell.color == "red" and cell.is_open
            for cell in self.world.values()
        ):
            self.red_opened = True
        if (
            any(cell.kind == "ball" for cell in self.world.values())
            and "pick up the" not in mission_lower
        ):
            self.dynamic_mode = True
        self.requires_key = "key" in self.mission.lower() or any(
            cell.kind == "door" and cell.is_locked for cell in self.world.values()
        )

    def _best_position(
        self,
        view: Grid,
        visible: np.ndarray,
        candidates: list[tuple[int, int]],
        direction: int,
    ) -> tuple[int, int]:
        """Choose the pose that best matches already mapped static landmarks."""
        width, height = visible.shape
        center_x = width // 2
        forward_x, forward_y = FORWARD[direction]
        right_x, right_y = -forward_y, forward_x
        scores: list[tuple[int, int, int, tuple[int, int]]] = []

        for candidate_index, candidate in enumerate(candidates):
            mismatches = 0
            matches = 0
            for view_x in range(width):
                for view_y in range(height):
                    if not visible[view_x, view_y]:
                        continue
                    if view_x == center_x and view_y == height - 1:
                        continue
                    observed = self._cell_from_obj(view.get(view_x, view_y))
                    if observed.kind == "ball":
                        continue
                    lateral = view_x - center_x
                    ahead = height - 1 - view_y
                    world_pos = (
                        candidate[0] + forward_x * ahead + right_x * lateral,
                        candidate[1] + forward_y * ahead + right_y * lateral,
                    )
                    known = self.world.get(world_pos)
                    if known is None or known.kind == "ball":
                        continue
                    if known.kind == observed.kind:
                        matches += 1
                    else:
                        mismatches += 1
            # Prefer the forward position if the view has no distinguishing landmarks.
            scores.append((mismatches, -matches, candidate_index, candidate))

        return min(scores)[3]

    def _search(self, is_goal: Callable[[SearchState], bool]) -> list[int] | None:
        opened = frozenset(
            pos for pos, cell in self.world.items() if cell.kind == "door" and cell.is_open
        )
        start = SearchState(*self.pos, self.direction, self.has_key, opened)
        queue = deque([start])
        parent: dict[SearchState, tuple[SearchState, int] | None] = {start: None}

        while queue:
            state = queue.popleft()
            if is_goal(state):
                actions: list[int] = []
                cursor = state
                while parent[cursor] is not None:
                    previous, action = parent[cursor]  # type: ignore[misc]
                    actions.append(action)
                    cursor = previous
                actions.reverse()
                return actions

            for action, next_state in self._successors(state):
                if next_state not in parent:
                    parent[next_state] = (state, action)
                    queue.append(next_state)
        return None

    def _successors(self, state: SearchState) -> Iterable[tuple[int, SearchState]]:
        yield int(Actions.left), SearchState(
            state.x, state.y, (state.direction - 1) % 4, state.has_key, state.opened_doors
        )
        yield int(Actions.right), SearchState(
            state.x, state.y, (state.direction + 1) % 4, state.has_key, state.opened_doors
        )

        front_pos = self._front(state.pos, state.direction)
        front_cell = self.world.get(front_pos)
        if front_cell is not None and self._walkable(front_cell, state.opened_doors, front_pos):
            yield int(Actions.forward), SearchState(
                *front_pos, state.direction, state.has_key, state.opened_doors
            )

        if front_cell is not None and front_cell.kind == "key" and not state.has_key:
            yield int(Actions.pickup), SearchState(
                state.x, state.y, state.direction, True, state.opened_doors
            )

        if front_cell is not None and front_cell.kind == "door" and front_pos not in state.opened_doors:
            blue_door_too_early = (
                self.red_blue_mode
                and not self.red_opened
                and front_cell.color == "blue"
            )
            if (not front_cell.is_locked or state.has_key) and not blue_door_too_early:
                yield int(Actions.toggle), SearchState(
                    state.x,
                    state.y,
                    state.direction,
                    state.has_key,
                    state.opened_doors | {front_pos},
                )

    def _explore_action(self) -> int:
        """Move to a reachable frontier, then step into its newly seen tile."""
        def at_frontier(state: SearchState) -> bool:
            front_pos = self._front(state.pos, state.direction)
            return front_pos not in self.world and self._walkable(
                self.world.get(state.pos, Cell("wall", can_overlap=False)),
                state.opened_doors,
                state.pos,
            )

        path = self._search(at_frontier)
        if path:
            return path[0]
        if path == []:
            return int(Actions.forward)
        # A harmless turn is a safe fallback for environments with a smaller
        # action space, such as Dynamic-Obstacles.
        return int(Actions.left)

    def _red_blue_action(self) -> int:
        """Open the red door first, then the blue door to finish the task."""
        target_color = "blue" if self.red_opened else "red"
        target_doors = [
            pos
            for pos, cell in self.world.items()
            if cell.kind == "door" and cell.color == target_color and not cell.is_open
        ]
        if not target_doors:
            return self._explore_action()

        path = self._search(
            lambda state: (
                (front := self._front(state.pos, state.direction)) in target_doors
                and front not in state.opened_doors
            )
        )
        if path:
            return path[0]
        if path == []:
            return int(Actions.toggle)
        return self._explore_action()

    def _clear_dynamic_obstacles(self) -> None:
        """Forget stale obstacle locations; they move after every action."""
        for pos, cell in list(self.world.items()):
            if cell.kind == "ball":
                self.world[pos] = Cell("empty")

    def _pickup_target_action(self) -> int | None:
        """Navigate to the object named by a pickup mission, then pick it up."""
        if self.target_kind is None:
            return None
        target_positions = [
            pos
            for pos, cell in self.world.items()
            if cell.kind == self.target_kind
            and (self.target_color is None or cell.color == self.target_color)
        ]
        if not target_positions:
            return None

        # MiniGrid permits carrying only one object. Keep the key until every
        # known locked door is open, then drop it on an empty tile before
        # collecting a box or ball.
        locked_doors_remain = any(
            cell.kind == "door" and cell.is_locked and not cell.is_open
            for cell in self.world.values()
        )
        if self.carrying_kind is not None and not locked_doors_remain:
            return self._drop_carried_item_action()

        path = self._search(
            lambda state: self._front(state.pos, state.direction) in target_positions
        )
        if path:
            return path[0]
        if path == []:
            return int(Actions.pickup)
        return None

    def _drop_carried_item_action(self) -> int | None:
        """Find a safe empty tile in front of the agent and drop its key."""
        def facing_empty(state: SearchState) -> bool:
            cell = self.world.get(self._front(state.pos, state.direction))
            return cell is not None and cell.kind == "empty"

        path = self._search(facing_empty)
        if path:
            return path[0]
        if path == []:
            return int(Actions.drop)
        return None

    def act(self, observation: dict) -> int:
        """Return one MiniGrid action for the current observation."""
        position_candidates = self._apply_previous_action()
        if self.dynamic_mode:
            self._clear_dynamic_obstacles()
        self._observe(observation, position_candidates)

        if self.red_blue_mode:
            action = self._red_blue_action()
            self._remember(action)
            return action

        if self.requires_key and not self.has_key:
            key_positions = [pos for pos, cell in self.world.items() if cell.kind == "key"]
            if key_positions:
                path = self._search(lambda state: state.has_key)
                if path:
                    action = path[0]
                    self._remember(action)
                    return action

        pickup_action = self._pickup_target_action()
        if pickup_action is not None:
            self._remember(pickup_action)
            return pickup_action

        goal_positions = [pos for pos, cell in self.world.items() if cell.kind == "goal"]
        if goal_positions:
            path = self._search(
                lambda state: state.pos in goal_positions
                and (not self.requires_key or state.has_key)
            )
            if path:
                action = path[0]
                self._remember(action)
                return action

        action = self._explore_action()
        self._remember(action)
        return action

    def _remember(self, action: int) -> None:
        front_pos = self._front(self.pos, self.direction)
        self.pending = (action, self.pos, self.direction, front_pos, self.world.get(front_pos))


def run_episode(
    env: gym.Env,
    agent: MiniGridAgent,
    seed: int | None = None,
    delay: float = 0.0,
) -> dict:
    observation, _ = env.reset(seed=seed)
    agent.reset()
    limit = getattr(env.unwrapped, "max_steps", 1000)

    total_reward = 0.0
    for step in range(1, limit + 1):
        action = agent.act(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if delay > 0:
            time.sleep(delay)
        if terminated or truncated:
            return {
                "success": bool(terminated and total_reward > 0),
                "steps": step,
                "reward": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

    return {
        "success": False,
        "steps": limit,
        "reward": total_reward,
        "terminated": False,
        "truncated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default="MiniGrid-Empty-8x8-v0",
        help=(
            "Gymnasium MiniGrid environment ID, such as MiniGrid-Empty-8x8-v0, "
            "MiniGrid-LavaGapS5-v0, MiniGrid-DoorKey-8x8-v0, "
            "MiniGrid-UnlockPickup-v0, or MiniGrid-KeyCorridorS3R1-v0"
        ),
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Show the environment window")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Pause in seconds after each action (use 0 for fastest execution)",
    )
    args = parser.parse_args()

    env = gym.make(args.env, render_mode="human" if args.render else None)
    agent = MiniGridAgent(args.env)
    try:
        for episode in range(args.episodes):
            result = run_episode(
                env,
                agent,
                seed=args.seed + episode,
                delay=args.delay if args.render else 0.0,
            )
            print(f"episode={episode + 1} {result}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
