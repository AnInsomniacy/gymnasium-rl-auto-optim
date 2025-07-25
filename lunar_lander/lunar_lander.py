import gymnasium as gym
import numpy as np
import json
import os
import signal
import sys
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner

GAME_ID = "LunarLander-v3"
CONTINUOUS = True
ALGORITHM_NAME = "ppo"
AUTO_DEVICE = False


def get_file_prefix():
    return f"{GAME_ID.replace('-', '_').lower()}_{ALGORITHM_NAME}"


def get_device():
    if not AUTO_DEVICE:
        print("Using CPU (auto device disabled)")
        return "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("Using CPU")
    return device


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_update_step = 0

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
                        self.episode_rewards)
                    print(
                        f"Episode {len(self.episode_rewards):4d} | Reward: {info['episode']['r']:7.2f} | Step: {info['episode']['l']:3d} | Avg100: {avg_reward:6.2f} | Total: {self.num_timesteps}")

        if self.num_timesteps - self.last_update_step >= 2048:
            self.last_update_step = self.num_timesteps
            updates = self.num_timesteps // 2048
            print(
                f"*** {ALGORITHM_NAME.upper()} Update {updates:3d} completed (Final Training) | Total: {self.num_timesteps} ***")
        return True


class HPOCallback(BaseCallback):
    def __init__(self, trial, eval_freq=10000, pruning_warmup=50000, verbose=0):
        super(HPOCallback, self).__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.pruning_warmup = pruning_warmup
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_eval_step = 0
        self.last_update_step = 0

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
                        self.episode_rewards)
                    print(
                        f"Episode {len(self.episode_rewards):4d} | Reward: {info['episode']['r']:7.2f} | Step: {info['episode']['l']:3d} | Avg100: {avg_reward:6.2f} | Total: {self.num_timesteps}")

        if self.num_timesteps - self.last_update_step >= 2048:
            self.last_update_step = self.num_timesteps
            updates = self.num_timesteps // 2048
            print(
                f"*** {ALGORITHM_NAME.upper()} Update {updates:3d} completed (Trial {self.trial.number + 1}) | Total: {self.num_timesteps} ***")

        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                if self.num_timesteps >= self.pruning_warmup:
                    self.trial.report(avg_reward, self.num_timesteps)
                    if self.trial.should_prune():
                        print(
                            f"Trial {self.trial.number + 1} pruned at step {self.num_timesteps} (avg: {avg_reward:.2f}) | Total: {self.num_timesteps}")
                        raise optuna.exceptions.TrialPruned()
                print(
                    f"Trial Progress: Step {self.num_timesteps:6d} | Episodes: {len(self.episode_rewards):4d} | Avg10: {avg_reward:6.2f} | Total: {self.num_timesteps}")
        return True


class RLAgent:
    def __init__(self, render_mode):
        self.render_mode = render_mode
        self.continuous = CONTINUOUS
        self.model = None
        self.env = None

    def create_env(self):
        env = gym.make(GAME_ID, continuous=self.continuous, render_mode=self.render_mode)
        env = Monitor(env)
        return DummyVecEnv([lambda: env])

    def save_emergency_final(self, signum=None, frame=None):
        if self.model:
            model_name = f"{get_file_prefix()}_final"
            self.model.save(model_name)
            print(f"\nEmergency save: Model saved to '{model_name}.zip'")
        if signum:
            if self.env:
                self.env.close()
            sys.exit(0)

    def train_with_params(self, params, total_timesteps, trial=None, pruning_warmup=50000, test_episodes=5):
        if trial is None:
            signal.signal(signal.SIGINT, self.save_emergency_final)

        self.env = self.create_env()
        device = get_device()

        if trial:
            callback = HPOCallback(trial, pruning_warmup=pruning_warmup)
            print(f"\nStarting Trial {trial.number + 1} training...")
        else:
            callback = TrainingCallback()
            print(f"\nStarting final model training...")

        policy_kwargs = {
            "net_arch": params['net_arch'],
            "activation_fn": nn.Tanh,
            "ortho_init": True
        }

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            gae_lambda=0.95,
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=0.5,
            max_grad_norm=0.5,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0
        )

        try:
            print(
                f"Training: {total_timesteps:,} steps | LR: {params['learning_rate']:.2e} | Arch: {params['net_arch']} | Starting...")
            self.model.learn(total_timesteps=total_timesteps, callback=callback)

            print(f"\nEvaluating with {test_episodes} test episodes | Total: {total_timesteps}...")
            total_rewards = []
            for i in range(test_episodes):
                obs = self.env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward[0]
                    if done:
                        total_rewards.append(episode_reward)
                        print(f"  Test {i + 1}/{test_episodes}: {episode_reward:.2f} | Total: {total_timesteps}")
                        break

            avg_reward = np.mean(total_rewards)
            if trial:
                print(f"Trial {trial.number + 1} completed | Score: {avg_reward:.2f} | Total: {total_timesteps}")
            else:
                print(f"Final training completed | Score: {avg_reward:.2f} | Total: {total_timesteps}")
            return avg_reward

        except optuna.exceptions.TrialPruned:
            print(f"Trial {trial.number + 1} was pruned early | Total: {self.model.num_timesteps if self.model else 0}")
            return -1000.0
        except KeyboardInterrupt:
            current_steps = self.model.num_timesteps if self.model else 0
            if trial is None:
                print(f"\nFinal training interrupted | Total: {current_steps}")
                if self.model:
                    model_name = f"{get_file_prefix()}_final"
                    self.model.save(model_name)
                    print(f"Model saved to '{model_name}.zip'")
            else:
                print(f"\nTrial {trial.number + 1} interrupted | Total: {current_steps}")
                raise
            return -1000.0
        finally:
            if self.env:
                self.env.close()


def objective(trial, hpo_config):
    net_arch_map = {
        "32x32": [32, 32],
        "64x64": [64, 64],
        "128x128": [128, 128],
        "256x256": [256, 256]
    }

    net_arch_str = trial.suggest_categorical('net_arch', ["32x32", "64x64", "128x128", "256x256"])
    n_steps = trial.suggest_categorical('n_steps', [256, 512, 1024, 2048])
    batch_size_raw = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])

    batch_size = min(batch_size_raw, n_steps)

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': trial.suggest_categorical('n_epochs', [3, 5, 10, 20]),
        'gamma': trial.suggest_float('gamma', 0.98, 0.999),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
        'net_arch': net_arch_map[net_arch_str],
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    }

    print(f"\n{'=' * 60}")
    print(f"Trial {trial.number + 1} - Testing Parameters")
    print(f"{'=' * 60}")
    print(
        f"LR: {params['learning_rate']:.2e} | Steps: {params['n_steps']} | Batch: {params['batch_size']} | Epochs: {params['n_epochs']}")
    if batch_size != batch_size_raw:
        print(f"Note: Batch size adjusted from {batch_size_raw} to {batch_size} (n_steps constraint)")
    print(
        f"Gamma: {params['gamma']:.3f} | Clip: {params['clip_range']:.2f} | Arch: {params['net_arch']} | Ent: {params['ent_coef']:.2e}")

    agent = RLAgent(render_mode=None)
    return agent.train_with_params(
        params,
        total_timesteps=hpo_config['hpo_timesteps'],
        trial=trial,
        pruning_warmup=hpo_config['pruning_warmup'],
        test_episodes=hpo_config['test_episodes']
    )


def run_hpo(hpo_config):
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    STUDY_FILE = f"{get_file_prefix()}_study.pkl"

    print(f"Starting HPO for {GAME_ID} {ALGORITHM_NAME.upper()}")
    print(f"Environment: {GAME_ID} | Continuous: {CONTINUOUS} | Algorithm: {ALGORITHM_NAME.upper()}")
    print(f"Config: {hpo_config['n_trials']} trials x {hpo_config['hpo_timesteps']:,} steps")
    print("Press Ctrl+C to interrupt and save progress")

    if os.path.exists(STUDY_FILE):
        with open(STUDY_FILE, "rb") as f:
            study = pickle.load(f)
        completed_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Continuing HPO: {completed_count} trials completed")
        if completed_count > 0:
            print(f"Current best score: {study.best_value:.2f}")

        remaining = max(0, hpo_config['n_trials'] - len(study.trials))
        if remaining == 0:
            print("All trials completed!")
            return study.best_params if completed_count > 0 else None
        print(f"Running {remaining} more trials")
    else:
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        print("Created new HPO study")
        remaining = hpo_config['n_trials']

    try:
        study.optimize(lambda trial: objective(trial, hpo_config), n_trials=remaining)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"\nHPO completed! Best score: {study.best_value:.2f}")
        print("Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        best_params_to_save = study.best_params.copy()
        net_arch_map = {
            "32x32": [32, 32],
            "64x64": [64, 64],
            "128x128": [128, 128],
            "256x256": [256, 256]
        }
        if 'net_arch' in best_params_to_save and isinstance(best_params_to_save['net_arch'], str):
            best_params_to_save['net_arch'] = net_arch_map[best_params_to_save['net_arch']]

        params_file = f"{get_file_prefix()}_params.json"
        with open(params_file, "w") as f:
            json.dump(best_params_to_save, f, indent=2)
        print(f"Best parameters saved to '{params_file}'")

    except KeyboardInterrupt:
        print(f"\nHPO interrupted!")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        total_trials = len(study.trials)

        if completed_trials:
            print(f"Progress: {len(completed_trials)}/{total_trials} trials completed")
            print(f"Current best score: {study.best_value:.2f}")
            print("Current best parameters:")
            for key, value in study.best_params.items():
                print(f"  {key}: {value}")

            best_params_to_save = study.best_params.copy()
            net_arch_map = {
                "32x32": [32, 32],
                "64x64": [64, 64],
                "128x128": [128, 128],
                "256x256": [256, 256]
            }
            if 'net_arch' in best_params_to_save and isinstance(best_params_to_save['net_arch'], str):
                best_params_to_save['net_arch'] = net_arch_map[best_params_to_save['net_arch']]

            params_file = f"{get_file_prefix()}_params.json"
            with open(params_file, "w") as f:
                json.dump(best_params_to_save, f, indent=2)
            print(f"Current best saved to '{params_file}'")
        else:
            print(f"No trials completed yet (interrupted during trial {total_trials})")
            if total_trials > 0:
                last_trial = study.trials[-1]
                print("Last trial parameters:")
                for key, value in last_trial.params.items():
                    print(f"  {key}: {value}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if completed_trials:
        clean_study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        for trial in completed_trials:
            clean_study.add_trial(trial)

        with open(STUDY_FILE, "wb") as f:
            pickle.dump(clean_study, f)
        print(f"Study saved with {len(completed_trials)} completed trials")
    else:
        if os.path.exists(STUDY_FILE):
            os.remove(STUDY_FILE)
        print("No completed trials - study file removed")

    return study.best_params if completed_trials else None


def train_with_best_params(final_timesteps=200000):
    FINAL_MODEL = f"{get_file_prefix()}_final"
    params_file = f"{get_file_prefix()}_params.json"

    if not os.path.exists(params_file):
        print(f"No {params_file} found. Run HPO first!")
        return

    with open(params_file, "r") as f:
        best_params = json.load(f)

    net_arch_map = {
        "32x32": [32, 32],
        "64x64": [64, 64],
        "128x128": [128, 128],
        "256x256": [256, 256]
    }

    if isinstance(best_params['net_arch'], str):
        best_params['net_arch'] = net_arch_map[best_params['net_arch']]
        print(f"Fixed net_arch format: {best_params['net_arch']}")

    agent = RLAgent(render_mode=None)
    device = get_device()

    if os.path.exists(f"{FINAL_MODEL}.zip"):
        print("Continuing training of existing model")
        agent.env = agent.create_env()
        agent.model = PPO.load(FINAL_MODEL, env=agent.env, device=device)

        try:
            print(f"Continuing training for {final_timesteps:,} additional steps...")
            callback = TrainingCallback()
            agent.model.learn(total_timesteps=final_timesteps, callback=callback)

            print(f"\nEvaluating continued training with 5 test episodes...")
            total_rewards = []
            for i in range(5):
                obs = agent.env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action, _ = agent.model.predict(obs, deterministic=True)
                    obs, reward, done, info = agent.env.step(action)
                    episode_reward += reward[0]
                    if done:
                        total_rewards.append(episode_reward)
                        print(f"  Test {i + 1}/5: {episode_reward:.2f} | Total: {final_timesteps}")
                        break

            final_score = np.mean(total_rewards)
            agent.model.save(FINAL_MODEL)
            print(f"Model updated! Score: {final_score:.2f} | Total: {final_timesteps}")

        except KeyboardInterrupt:
            print(f"Training interrupted | Total: {final_timesteps}")
            if agent.model:
                agent.model.save(FINAL_MODEL)
                print(f"Model saved to '{FINAL_MODEL}.zip'")
        finally:
            if agent.env:
                agent.env.close()
    else:
        print("Training new model with best parameters")
        print("Parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        try:
            final_score = agent.train_with_params(best_params, total_timesteps=final_timesteps)
            agent.model.save(FINAL_MODEL)
            print(f"Final model saved! Score: {final_score:.2f} | Total: {final_timesteps}")
        except KeyboardInterrupt:
            print(f"Training interrupted | Total: {final_timesteps}")
            if agent.model:
                agent.model.save(FINAL_MODEL)
                print(f"Model saved to '{FINAL_MODEL}.zip'")


def test_model():
    model_file = f"{get_file_prefix()}_final.zip"

    if not os.path.exists(model_file):
        print(f"No {model_file} found. Train first!")
        return

    print("Testing model...")

    agent = RLAgent(render_mode="human")
    agent.env = agent.create_env()
    device = get_device()
    agent.model = PPO.load(f"{get_file_prefix()}_final", device=device)

    total_rewards = []
    successes = 0

    for episode in range(10):
        obs = agent.env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, done, info = agent.env.step(action)
            episode_reward += reward[0]

            if done:
                total_rewards.append(episode_reward)
                if episode_reward > 200:
                    successes += 1
                    status = "SUCCESS"
                elif episode_reward > 100:
                    status = "GOOD"
                elif episode_reward > 0:
                    status = "OK"
                else:
                    status = "CRASH"

                print(f"Test {episode + 1:2d}/10 | Score: {episode_reward:7.2f} | {status}")
                break

    print(f"\nResults: Avg: {np.mean(total_rewards):.2f} | Success: {successes}/10 ({100 * successes / 10:.1f}%)")
    agent.env.close()


def main():
    HPO_CONFIG = {
        'n_trials': 50,
        'hpo_timesteps': 100000,
        'pruning_warmup': 50000,
        'test_episodes': 5
    }

    FINAL_TIMESTEPS = 200000

    print(f"{GAME_ID} {ALGORITHM_NAME.upper()} + Optuna HPO")
    print("=" * 40)
    print(f"Environment: {GAME_ID} | Continuous: {CONTINUOUS} | Algorithm: {ALGORITHM_NAME.upper()}")
    print(f"Auto Device: {AUTO_DEVICE}")
    print("Configuration:")
    for key, value in HPO_CONFIG.items():
        print(f"  {key}: {value}")
    print(f"  final_timesteps: {FINAL_TIMESTEPS:,}")
    print("=" * 40)
    print(f"\n{ALGORITHM_NAME.upper()} Parameter Ranges (RL Zoo Standards):")
    print("  learning_rate: 1e-5 to 1e-1 (log)")
    print("  ent_coef: 1e-8 to 1e-2 (log)")
    print("  batch_size: [32, 64, 128, 256, 512]")
    print("=" * 40)

    print("\nSelect Mode:")
    print("  1 - Run HPO")
    print("  2 - Train with best params")
    print("  3 - Test model")
    print("  4 - Exit")

    while True:
        mode = input("\nChoice [1-4]: ").strip()

        if mode == "1":
            print("Starting HPO...")
            run_hpo(HPO_CONFIG)
            break
        elif mode == "2":
            print("Training final model...")
            train_with_best_params(FINAL_TIMESTEPS)
            break
        elif mode == "3":
            test_model()
            break
        elif mode == "4":
            print("Exit")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
