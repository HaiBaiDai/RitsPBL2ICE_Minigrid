#

import argparse
import json
import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO


# MiniGrid action mapping
ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}


def evaluate_and_log(
    env_id,
    model_path,
    log_path,
    num_seeds,
    step_cap,
):
    print("====================================================")
    print(f"Evaluating PPO Agent on {env_id}")
    print(f"Model File  : {model_path}.zip")
    print(f"Seeds       : 0 to {num_seeds - 1}")
    print(f"Step Cap    : {step_cap} steps per episode")
    print("====================================================")

    # 1. Load trained Stable-Baselines3 model
    model = PPO.load(model_path)

    # 2. Append evaluation logs to JSONL file
    with open(log_path, "a", encoding="utf-8") as f:
        for seed in range(num_seeds):
            # Enforce partial observability and fixed seeds
            env = gym.make(env_id)
            env = FlatObsWrapper(env)
            obs, _ = env.reset(seed=seed)

            run_id = f"{env_id}_s{seed}"
            total_steps = 0
            solved = False

            # Enforce step cap
            for step in range(step_cap):
                total_steps += 1

                # Deterministic prediction from policy
                action_int, _ = model.predict(
                    obs,
                    deterministic=True,
                )
                action_int = int(action_int)
                action_name = ACTION_NAMES.get(
                    action_int,
                    "unknown",
                )

                next_obs, reward, terminated, truncated, _ = (
                    env.step(action_int)
                )
                done = terminated or truncated

                # Log step details
                step_record = {
                    "run_id": run_id,
                    "seed": seed,
                    "level": env_id,
                    "modality": "symbolic",
                    "step": step,
                    "state": str(obs.tolist()[:10]) + "...",
                    "action_raw": str(action_int),
                    "action": action_name,
                    "reward": float(reward),
                    "done": bool(done),
                }
                f.write(json.dumps(step_record) + "\n")

                obs = next_obs

                if done:
                    solved = bool(reward > 0)
                    break

            # Log final episode summary
            summary_record = {
                "run_id": run_id,
                "success": solved,
                "steps": total_steps,
                "prompt_tokens": 0,       # 0 for PPO
                "completion_tokens": 0,   # 0 for PPO
                "cost_usd": 0.0,          # 0.0 for PPO
            }
            f.write(json.dumps(summary_record) + "\n")

            env.close()

    print(
        f"Evaluation complete. Results appended to: "
        f"{log_path}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PPO Agent and Generate JSONL Logs"
    )

    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-DoorKey-8x8-v0",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ppo_doorkey",
        help="Path to saved .zip model",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="pilot_logs.jsonl",
        help="Output JSONL log file",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=40,
        help="Number of fixed seeds to evaluate",
    )
    parser.add_argument(
        "--step_cap",
        type=int,
        default=50,
        help="Maximum step cap per episode",
    )

    args = parser.parse_args()

    evaluate_and_log(
        env_id=args.env,
        model_path=args.model_path,
        log_path=args.log_path,
        num_seeds=args.num_seeds,
        step_cap=args.step_cap,
    )
