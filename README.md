# Gym Environment and DQN Training for an Advanced Connect4 Variant

The objective of this project is to develop a gym-like environment for an advanced variation of Connect4, called _4 in a Row - Evolution_, and to design and train a Deep Q-Network (DQN) model to play the game. The trained model was evaluated against a random opponent augmented with a heuristic strategy. Initially, the untrained model achieved approximately a 20% win rate and a 70% loss rate against the opponent. However, after around 100,000 training episodes, our best-performing model - a **Double Dueling DQN** with two convolutional layers and approximately 950,000 parameters - attained a 82% win rate and a 10% loss rate. These results demonstrate a promising initial performance, although a human player would likely achieve a higher win rate against this specific opponent.

## Code

To replicate this study, follow the instruction in the notebook `Train_4inarowEvolution.ipynb`.
