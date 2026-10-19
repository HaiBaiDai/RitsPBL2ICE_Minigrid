import argparse
import json
from collections import defaultdict


def compute_pilot_metrics(log_path="pilot_logs.jsonl"):
    # Dictionary to hold stats per environment:
    # level -> {total, solved, solved_steps}
    env_stats = defaultdict(
        lambda: {
            "total": 0,
            "solved": 0,
            "solved_steps": [],
        }
    )

    try:
        with open(log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                data = json.loads(line)

                # Process final summary lines (lines containing "success")
                if "success" in data:
                    run_id = data.get("run_id", "")

                    # Extract environment name from level field or run_id
                    # Example: "MiniGrid-DoorKey-8x8-v0_s0"
                    level = data.get("level")

                    if not level and "_s" in run_id:
                        level = run_id.rsplit("_s", 1)[0]
                    elif not level:
                        level = "Unknown Environment"

                    env_stats[level]["total"] += 1

                    if data["success"]:
                        env_stats[level]["solved"] += 1

                        # Save step count for solved runs only
                        env_stats[level]["solved_steps"].append(
                            data["steps"]
                        )

    except FileNotFoundError:
        print(f"Error: Log file '{log_path}' not found!")
        return

    if not env_stats:
        print(
            f"No summary records found in '{log_path}'. "
            "Did you run evaluate_ppo.py?"
        )
        return

    # Print formatted results table for Report 03
    print("\n" + "=" * 80)
    print("PPO BASELINE PILOT RESULTS TABLE (GROUPED BY ENVIRONMENT)")
    print("=" * 80)

    print(
        f"{'Environment Level':<35} | "
        f"{'Episodes':<6} | "
        f"{'Solved':<6} | "
        f"{'Success Rate (%)':<16} | "
        f"{'Mean Steps (Solved)':<20}"
    )
    print("-" * 80)

    for level, stats in env_stats.items():
        total = stats["total"]
        solved = stats["solved"]

        success_rate = (
            (solved / total * 100)
            if total > 0
            else 0.0
        )

        # Mean steps-to-goal averaged over solved runs only
        mean_steps = (
            sum(stats["solved_steps"])
            / len(stats["solved_steps"])
            if stats["solved_steps"]
            else 0.0
        )

        print(
            f"{level:<35} | "
            f"{total:<6} | "
            f"{solved:<6} | "
            f"{success_rate:<16.2f} | "
            f"{mean_steps:<20.2f}"
        )

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Pilot Results from JSONL Logs"
    )

    parser.add_argument(
        "--log_path",
        type=str,
        default="pilot_logs.jsonl",
        help="Path to JSONL log file",
    )

    args = parser.parse_args()
    compute_pilot_metrics(args.log_path)
