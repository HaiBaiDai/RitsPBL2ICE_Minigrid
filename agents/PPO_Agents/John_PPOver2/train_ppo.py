#PBL2 week4 onwards
#this PPO agent trains 5 of the team-elected minigrid environments of different difficulties.
#This PPO from StableBaselines3 is default and not customised or tuned. 
#FlatObsWrapper is added to ensure partial observability of PPO agent. 
#

import argparse
import gymnasium as gym
import minigrid
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO


def train_ppo(env_id, total_timesteps, save_path):
    print("====================================================")
    print(f"Starting PPO Training on {env_id}")
    print(f"Step Budget : {total_timesteps:,} steps")
    print(f"Saving to   : {save_path}.zip")
    print("====================================================")

    # 1. Create environment and wrap to partial 7x7 flat observation window
    env = gym.make(env_id)
    env = FlatObsWrapper(env)

    # 2. Instantiate Stable-Baselines3 PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )

    # 3. Train policy on step budget
    model.learn(total_timesteps=total_timesteps)

    # 4. Save trained model checkpoint
    model.save(save_path)
    print(f"\nTraining complete. Model saved to: {save_path}.zip\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO on any MiniGrid Environment"
    )

    # Command-line flags replacing hardcoded strings
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-DoorKey-8x8-v0",
        help="Gymnasium environment ID",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="ppo_doorkey",
        help="Output filename for saved .zip model",
    )

    args = parser.parse_args()

    train_ppo(
        env_id=args.env,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
    )
