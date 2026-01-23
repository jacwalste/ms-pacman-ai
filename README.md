# Atari AI - Rainbow DQN

A reinforcement learning agent that learns to play Atari games from scratch using the Rainbow DQN algorithm.

The agent sees only raw pixels and a score - no knowledge of game rules, enemies, or objectives. Through trial and error over thousands of games, it learns to play.

## Supported Games

Works with any Atari game. Pre-trained models included for:
- **Pong** - Deterministic, agent masters this one
- **MS Pac-Man** - Stochastic ghost behavior makes it harder
- **Breakout** - Learns the tunnel-through-the-side strategy
- **Space Invaders** - Learns to use cover and prioritize targets

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install torch gymnasium ale-py numpy
```

## Usage

### Watch a trained agent play

```bash
# Watch the best model play MS Pac-Man
python watch_game.py --game MsPacman

# Watch Pong (the agent dominates this one)
python watch_game.py --game Pong

# Watch a specific checkpoint
python watch_game.py --game MsPacman --model models/mspacman/ep1000.pth

# Watch multiple games
python watch_game.py --game Pong --games 5

# Compare untrained vs trained
python watch_game.py --game Pong --untrained --games 1  # Random flailing
python watch_game.py --game Pong --games 1               # Trained agent

# List all available models
python watch_game.py --list
```

### Train a new agent

```bash
# Train on a game
python train_game.py --game Pong --episodes 5000

# Resume training from checkpoint
python train_game.py --game MsPacman --episodes 10000 --resume
```

Training saves checkpoints automatically:
- `best.pth` - Best performing model (by 100-episode average)
- `checkpoint.pth` - Latest checkpoint (every 500 episodes)
- `ep{N}.pth` - Milestone saves at 500, 1000, 2000, 3000, 5000, 7500, 10000 episodes

## Project Structure

```
├── rainbow_agent.py          # Main RL agent (n-step, prioritized replay)
├── rainbow_network.py        # Neural network (dueling + distributional)
├── noisy_linear.py           # Noisy layers for exploration
├── prioritized_replay_buffer.py  # Sum-tree based prioritized replay
├── wrappers.py               # Environment preprocessing
├── train_game.py             # Training script
├── watch_game.py             # Demo/evaluation script
└── models/                   # Saved model checkpoints
    ├── pong/
    ├── mspacman/
    ├── breakout/
    └── spaceinvaders/
```

## The Algorithm

Rainbow DQN combines six improvements over vanilla DQN:

1. **Double DQN** - Reduces overestimation of action values
2. **Prioritized Replay** - Learns more from surprising experiences
3. **Dueling Networks** - Separates state value from action advantage
4. **Distributional RL (C51)** - Predicts distribution of returns, not just mean
5. **Noisy Networks** - Learned exploration (no epsilon-greedy)
6. **N-step Returns** - Uses 3-step lookahead for better credit assignment

## Hardware

- **Training**: GPU recommended (CUDA or Apple MPS). CPU works but is slow.
- **Watching**: CPU is fine, runs in real-time.

The code auto-detects available hardware (CUDA > MPS > CPU).

## References

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) (Hessel et al., 2017)
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (Mnih et al., 2015)
