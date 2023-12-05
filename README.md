My solutions to the programming assignments on Multi-Armed Bandits in the course CS747: Foundations of Intelligent and Learning Agents for Autumn 2023
# Faulty Bandits
Implemented optimal algorithms for Multi-Armed Bandits such as UCB, KL-UCB, and Thompson Sampling. Designed an asymptotically optimal algorithm to minimize the expected cumulative regret for bandit instance where pulls are no longer guaranteed to be successful and have a probability of giving faulty outputs sampled from a uniform distribution.
[[Report]](https://vansh28kapoor.github.io/assets/pdf/Bandits.pdf)
# Batched Multi-Armed Bandits
An efficient algorithm for batched sampling, allowing the pulling of only a fixed number of arms at each time-step, was required for this task. An asymptotically optimal algorithm was devised, which generalizes for arbitrary batch sizes.
