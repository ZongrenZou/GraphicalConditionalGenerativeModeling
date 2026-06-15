## Lunar Lander Evaluation Code

This directory contains the code used for the Lunar Lander control experiment in the paper.


The experiment uses Lunar Lander as a standard downstream control benchmark to evaluate whether the discovered ancestor sets are sufficient for policy training and transfer. The control/RL components are not methodological contributions of this work.

The learned flow-matching surrogate replaces only the thrust/impulse stage of the Gymnasium `LunarLander-v3` environment. The remaining simulator components, including Box2D stepping, contact handling, state extraction, reward shaping, and termination logic, follow the Gymnasium Lunar Lander environment. 

The SAC policies are trained using Stable-Baselines3. We use the continuous-action `LunarLander-v3` environment from Gymnasium and use SAC hyperparameters based on the Lunar Lander settings from RL Baselines3 Zoo.

The full-surrogate and pruned-surrogate experiments differ only in the conditioning variables supplied to the learned flow-matching impulse models. No changes are made to the proposed structure-discovery method in this directory; this code is used only for downstream policy-training and transfer evaluation.

## References and dependencies

This experiment uses:

- [Gymnasium](https://gymnasium.farama.org/) / Box2D `LunarLander-v3` for the true environment and simulator components.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for SAC policy training and evaluation utilities.
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) for the Lunar Lander SAC hyperparameter settings.

Please cite the corresponding Gymnasium, Stable-Baselines3, RL Baselines3 Zoo, SAC, and software references when using this code.
