import gymnasium as gym
import numpy as np
import json
import os
import signal
import sys
import pickle
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner
from tqdm import tqdm


def get_file_prefix(config):
    return f"{config['game_id'].replace('-', '_').lower()}_{config['algorithm_name']}"


def get_device(config):
    if not config['auto_device']:
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
    def __init__(self, pbar, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.pbar = pbar

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
                        self.episode_rewards)
                    self.pbar.set_description(
                        f"Training | Episodes: {len(self.episode_rewards)} | Avg100: {avg_reward:.2f}")
                    self.pbar.set_postfix({"Last": f"{info['episode']['r']:.1f}", "Steps": info['episode']['l']})

        self.pbar.update(1)
        return True


class HPOCallback(BaseCallback):
    def __init__(self, trial, pbar, eval_freq=10000, pruning_warmup=50000, verbose=0):
        super(HPOCallback, self).__init__(verbose)
        self.trial = trial
        self.eval_freq = eval_freq
        self.pruning_warmup = pruning_warmup
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_eval_step = 0
        self.pbar = pbar

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(
                        self.episode_rewards)
                    self.pbar.set_description(
                        f"Trial {self.trial.number + 1} | Episodes: {len(self.episode_rewards)} | Avg100: {avg_reward:.2f}")
                    self.pbar.set_postfix({"Last": f"{info['episode']['r']:.1f}", "Steps": info['episode']['l']})

        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.last_eval_step = self.num_timesteps
            if len(self.episode_rewards) >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                if self.num_timesteps >= self.pruning_warmup:
                    self.trial.report(avg_reward, self.num_timesteps)
                    if self.trial.should_prune():
                        self.pbar.close()
                        raise optuna.exceptions.TrialPruned()

        self.pbar.update(1)
        return True


class RLAgent:
    def __init__(self, render_mode, config):
        self.render_mode = render_mode
        self.config = config
        self.model = None
        self.env = None

    def create_env(self):
        env = gym.make(
            self.config['game_id'],
            render_mode=self.render_mode,
            max_episode_steps=self.config['max_episode_steps']
        )
        env = Monitor(env)
        return DummyVecEnv([lambda: env])

    def save_emergency_final(self, signum=None, frame=None):
        if self.model:
            model_name = f"{get_file_prefix(self.config)}_final"
            self.model.save(model_name)
            print(f"\nEmergency save: Model saved to '{model_name}.zip'")
        if signum:
            if self.env:
                self.env.close()
            sys.exit(0)

    def train_with_params(self, params, total_timesteps, trial=None, pruning_warmup=50000, test_episodes=5):
        pbar = None
        try:
            if trial is None:
                signal.signal(signal.SIGINT, self.save_emergency_final)

            self.env = self.create_env()
            device = get_device(self.config)

            policy_kwargs = {
                "net_arch": params['net_arch'],
                "activation_fn": nn.ReLU,
            }

            self.model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=params['learning_rate'],
                buffer_size=params['buffer_size'],
                learning_starts=params['learning_starts'],
                batch_size=params['batch_size'],
                tau=params['tau'],
                gamma=params['gamma'],
                train_freq=params['train_freq'],
                gradient_steps=params['gradient_steps'],
                target_update_interval=params['target_update_interval'],
                exploration_fraction=params['exploration_fraction'],
                exploration_initial_eps=params['exploration_initial_eps'],
                exploration_final_eps=params['exploration_final_eps'],
                max_grad_norm=10,
                policy_kwargs=policy_kwargs,
                device=device,
                verbose=0
            )

            if trial:
                pbar = tqdm(total=total_timesteps, desc=f"Trial {trial.number + 1}", unit="step")
                callback = HPOCallback(trial, pbar, pruning_warmup=pruning_warmup)
            else:
                pbar = tqdm(total=total_timesteps, desc="Training", unit="step")
                callback = TrainingCallback(pbar)
                print(f"\nStarting final model training...")

            self.model.learn(total_timesteps=total_timesteps, callback=callback)
            pbar.close()

            print(f"\nEvaluating with {test_episodes} test episodes...")
            total_rewards = []
            test_pbar = tqdm(range(test_episodes), desc="Testing", unit="episode")
            for i in test_pbar:
                obs = self.env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward[0]
                    if done:
                        total_rewards.append(episode_reward)
                        test_pbar.set_postfix({"Score": f"{episode_reward:.2f}"})
                        break
            test_pbar.close()

            avg_reward = np.mean(total_rewards)
            if trial:
                print(f"Trial {trial.number + 1} completed | Score: {avg_reward:.2f}")
            else:
                print(f"Final training completed | Score: {avg_reward:.2f}")
            return avg_reward

        except optuna.exceptions.TrialPruned:
            if pbar:
                pbar.close()
            print(f"Trial {trial.number + 1} was pruned early")
            return -1000.0
        except KeyboardInterrupt:
            if pbar:
                pbar.close()
            current_steps = self.model.num_timesteps if self.model else 0
            if trial is None:
                print(f"\nFinal training interrupted at step {current_steps}")
                if self.model:
                    model_name = f"{get_file_prefix(self.config)}_final"
                    self.model.save(model_name)
                    print(f"Model saved to '{model_name}.zip'")
            else:
                print(f"\nTrial {trial.number + 1} interrupted at step {current_steps}")
                raise
            return -1000.0
        finally:
            if pbar:
                pbar.close()
            if self.env:
                self.env.close()


def objective(trial, config):
    net_arch_map = {
        "64": [64],
        "128": [128],
        "256": [256],
        "64x64": [64, 64],
        "128x128": [128, 128],
        "256x256": [256, 256]
    }

    net_arch_str = trial.suggest_categorical('net_arch', ["64", "128", "256", "64x64", "128x128", "256x256"])

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000, 500000]),
        'learning_starts': trial.suggest_categorical('learning_starts', [1000, 5000, 10000]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'tau': trial.suggest_float('tau', 0.001, 0.1, log=True),
        'gamma': trial.suggest_float('gamma', 0.95, 0.999),
        'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8, 16]),
        'gradient_steps': trial.suggest_categorical('gradient_steps', [1, 2, 4]),
        'target_update_interval': trial.suggest_categorical('target_update_interval', [1000, 5000, 10000]),
        'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
        'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.5, 1.0),
        'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
        'net_arch': net_arch_map[net_arch_str]
    }

    study = trial.study
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f"\n{'=' * 60}")
    print(f"Trial {trial.number + 1} - Testing Parameters")
    if completed_trials:
        best_trial = max(completed_trials, key=lambda t: t.value)
        print(f"Best So Far: Trial {best_trial.number + 1} - Score: {best_trial.value:.2f}")
    else:
        print(f"Best So Far: No previous trials")
    print(f"{'=' * 60}")
    print(
        f"LR: {params['learning_rate']:.2e} | Buffer: {params['buffer_size']} | Batch: {params['batch_size']} | Tau: {params['tau']:.3f}")
    print(
        f"Gamma: {params['gamma']:.3f} | Train Freq: {params['train_freq']} | Arch: {params['net_arch']} | Eps: {params['exploration_final_eps']:.3f}")

    agent = RLAgent(render_mode=None, config=config)
    return agent.train_with_params(
        params,
        total_timesteps=config['hpo_timesteps'],
        trial=trial,
        pruning_warmup=config['pruning_warmup'],
        test_episodes=config['hpo_test_episodes']
    )


def save_best_params(study, config):
    best_params_to_save = study.best_params.copy()

    net_arch_map = {
        "64": [64],
        "128": [128],
        "256": [256],
        "64x64": [64, 64],
        "128x128": [128, 128],
        "256x256": [256, 256]
    }
    if 'net_arch' in best_params_to_save and isinstance(best_params_to_save['net_arch'], str):
        best_params_to_save['net_arch'] = net_arch_map[best_params_to_save['net_arch']]

    params_file = f"{get_file_prefix(config)}_params.json"
    with open(params_file, "w") as f:
        json.dump(best_params_to_save, f, indent=2)
    print(f"Best parameters saved to '{params_file}'")


def run_hpo(config):
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    STUDY_FILE = f"{get_file_prefix(config)}_study.pkl"

    print(f"Starting HPO for {config['game_id']} {config['algorithm_name'].upper()}")
    print(f"Environment: {config['game_id']} | Algorithm: {config['algorithm_name'].upper()}")
    print(f"Config: {config['n_trials']} trials x {config['hpo_timesteps']:,} steps")
    print("Press Ctrl+C to interrupt and save progress")

    if os.path.exists(STUDY_FILE):
        with open(STUDY_FILE, "rb") as f:
            study = pickle.load(f)
        completed_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Continuing HPO: {completed_count} trials completed")
        if completed_count > 0:
            print(f"Current best score: {study.best_value:.2f}")

        remaining = max(0, config['n_trials'] - len(study.trials))
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
        remaining = config['n_trials']

    try:
        study.optimize(lambda trial: objective(trial, config), n_trials=remaining)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        best_trial = max(completed_trials, key=lambda t: t.value)
        print(f"\nHPO completed! Best score: {study.best_value:.2f} (Trial {best_trial.number + 1})")
        print("Best parameters:")

        best_params = study.best_params
        best_arch = best_params['net_arch']
        if isinstance(best_arch, str):
            net_arch_map = {"64": [64], "128": [128], "256": [256], "64x64": [64, 64], "128x128": [128, 128],
                            "256x256": [256, 256]}
            best_arch = net_arch_map[best_arch]

        print(
            f"LR: {best_params['learning_rate']:.2e} | Buffer: {best_params['buffer_size']} | Batch: {best_params['batch_size']} | Tau: {best_params['tau']:.3f}")
        print(
            f"Gamma: {best_params['gamma']:.3f} | Train Freq: {best_params['train_freq']} | Arch: {best_arch} | Eps: {best_params['exploration_final_eps']:.3f}")

        save_best_params(study, config)

    except KeyboardInterrupt:
        print(f"\nHPO interrupted!")

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        total_trials = len(study.trials)

        if completed_trials:
            best_trial = max(completed_trials, key=lambda t: t.value)
            print(f"Progress: {len(completed_trials)}/{total_trials} trials completed")
            print(f"Current best score: {study.best_value:.2f} (Trial {best_trial.number + 1})")
            print("Current best parameters:")

            best_params = study.best_params
            best_arch = best_params['net_arch']
            if isinstance(best_arch, str):
                net_arch_map = {"64": [64], "128": [128], "256": [256], "64x64": [64, 64], "128x128": [128, 128],
                                "256x256": [256, 256]}
                best_arch = net_arch_map[best_arch]

            print(
                f"LR: {best_params['learning_rate']:.2e} | Buffer: {best_params['buffer_size']} | Batch: {best_params['batch_size']} | Tau: {best_params['tau']:.3f}")
            print(
                f"Gamma: {best_params['gamma']:.3f} | Train Freq: {best_params['train_freq']} | Arch: {best_arch} | Eps: {best_params['exploration_final_eps']:.3f}")

            save_best_params(study, config)
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


def train_with_best_params(config):
    FINAL_MODEL = f"{get_file_prefix(config)}_final"
    params_file = f"{get_file_prefix(config)}_params.json"

    if not os.path.exists(params_file):
        print(f"No {params_file} found. Run HPO first!")
        return

    with open(params_file, "r") as f:
        best_params = json.load(f)

    net_arch_map = {
        "64": [64],
        "128": [128],
        "256": [256],
        "64x64": [64, 64],
        "128x128": [128, 128],
        "256x256": [256, 256]
    }

    if isinstance(best_params['net_arch'], str):
        best_params['net_arch'] = net_arch_map[best_params['net_arch']]
        print(f"Fixed net_arch format: {best_params['net_arch']}")

    agent = RLAgent(render_mode=None, config=config)
    device = get_device(config)

    final_timesteps = config['final_timesteps']

    if os.path.exists(f"{FINAL_MODEL}.zip"):
        print("Continuing training of existing model")
        agent.env = agent.create_env()
        agent.model = DQN.load(FINAL_MODEL, env=agent.env, device=device)

        pbar = None
        try:
            print(f"Continuing training for {final_timesteps:,} additional steps...")
            pbar = tqdm(total=final_timesteps, desc="Training", unit="step")
            callback = TrainingCallback(pbar)
            agent.model.learn(total_timesteps=final_timesteps, callback=callback)
            pbar.close()

            print(f"\nEvaluating continued training with 5 test episodes...")
            total_rewards = []
            test_pbar = tqdm(range(5), desc="Testing", unit="episode")
            for i in test_pbar:
                obs = agent.env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action, _ = agent.model.predict(obs, deterministic=True)
                    obs, reward, done, info = agent.env.step(action)
                    episode_reward += reward[0]
                    if done:
                        total_rewards.append(episode_reward)
                        test_pbar.set_postfix({"Score": f"{episode_reward:.2f}"})
                        break
            test_pbar.close()

            final_score = np.mean(total_rewards)
            agent.model.save(FINAL_MODEL)
            print(f"Model updated! Score: {final_score:.2f}")

        except KeyboardInterrupt:
            if pbar:
                pbar.close()
            print(f"Training interrupted")
            if agent.model:
                agent.model.save(FINAL_MODEL)
                print(f"Model saved to '{FINAL_MODEL}.zip'")
        finally:
            if pbar:
                pbar.close()
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
            print(f"Final model saved! Score: {final_score:.2f}")
        except KeyboardInterrupt:
            print(f"Training interrupted")
            if agent.model:
                agent.model.save(FINAL_MODEL)
                print(f"Model saved to '{FINAL_MODEL}.zip'")


def test_model(config):
    model_file = f"{get_file_prefix(config)}_final.zip"

    if not os.path.exists(model_file):
        print(f"No {model_file} found. Train first!")
        return

    print(f"Testing model with {config['test_episodes']} episodes...")

    agent = RLAgent(render_mode="human", config=config)
    agent.env = agent.create_env()
    device = get_device(config)
    agent.model = DQN.load(f"{get_file_prefix(config)}_final", device=device)

    total_rewards = []
    successes = 0

    test_pbar = tqdm(range(config['test_episodes']), desc="Testing", unit="episode")
    for episode in test_pbar:
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

                test_pbar.set_postfix({"Score": f"{episode_reward:.2f}", "Status": status})
                break
    test_pbar.close()

    print(
        f"\nResults: Avg: {np.mean(total_rewards):.2f} | Success: {successes}/{config['test_episodes']} ({100 * successes / config['test_episodes']:.1f}%)")
    agent.env.close()


def main():
    CONFIG = {
        'game_id': "LunarLander-v3",
        'algorithm_name': "dqn",
        'auto_device': False,
        'max_episode_steps': 1000,
        'n_trials': 50,
        'hpo_timesteps': 2000,
        'pruning_warmup': 500,
        'hpo_test_episodes': 50,
        'final_timesteps': 2000000,
        'test_episodes': 20
    }

    print(f"Environment: {CONFIG['game_id']} | Algorithm: {CONFIG['algorithm_name'].upper()}")
    print(f"Auto Device: {CONFIG['auto_device']}")

    print("\nSelect Mode:")
    print("  1 - Run HPO")
    print("  2 - Train with best params")
    print("  3 - Test model")
    print("  4 - Exit")

    while True:
        mode = input("\nChoice [1-4]: ").strip()

        if mode == "1":
            print("Starting HPO...")
            run_hpo(CONFIG)
            break
        elif mode == "2":
            print("Training final model...")
            train_with_best_params(CONFIG)
            break
        elif mode == "3":
            test_model(CONFIG)
            break
        elif mode == "4":
            print("Exit")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
