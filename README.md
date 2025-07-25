# gymnasium-rl-auto-optim

Universal reinforcement learning framework with automatic hyperparameter optimization for all Gymnasium environments.

## Features

- **Auto Hyperparameter Optimization**: Uses Optuna to find optimal parameters automatically
- **Multi-Algorithm Support**: DQN, PPO, and other Stable-Baselines3 algorithms
- **Universal Environment Compatibility**: Works with all Gymnasium environments
- **Intelligent Training Pipeline**: HPO → Best params training → Model testing
- **Resume Training**: Interrupt and continue training with state preservation
- **Smart Pruning**: Early termination of poor-performing trials

## Supported Algorithms

- **DQN**: For discrete action spaces (LunarLander, CartPole, etc.)
- **PPO**: For continuous action spaces (CarRacing, BipedalWalker, etc.)
- **Auto-detection**: Automatically selects appropriate algorithm based on environment
