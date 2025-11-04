# Oware AI Competition Bot Game

A comprehensive Oware (Awale) AI competition framework featuring multiple AI agents, training capabilities, tournaments, and statistical analysis. This project implements the traditional African board game Oware with various AI strategies and deep reinforcement learning agents.

## Table of Contents

- [About Oware](#about-oware)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Game Rules](#game-rules)
- [AI Agents](#ai-agents)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)

## About Oware

Oware is a traditional African board game from the Mancala family. Two players compete to capture seeds by strategically moving them around a board with 12 pits (6 per player). The game combines tactical thinking with strategic planning, making it an excellent testbed for AI algorithms.

## Features

- 🎯 **Multiple AI Strategies**: From simple random moves to sophisticated deep Q-networks
- 🧠 **Reinforcement Learning**: PyTorch-based DQN agents with experience replay
- 🏆 **Tournament System**: Single-elimination tournaments with configurable match formats
- 📊 **Statistical Analysis**: Comprehensive performance metrics and CSV logging
- ⚙️ **Game Variants**: Support for different rule variants and board configurations
- 🎮 **Interactive Menu**: Easy-to-use command-line interface
- 💾 **Model Persistence**: Save and load trained AI models

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Pandas (for analysis features)
- PyTorch (optional, for DQN agents)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jrain-dev/Oware-AI-Competition-Bot-Game.git
cd Oware-AI-Competition-Bot-Game
```

2. Install dependencies:
```bash
pip install numpy pandas
```

3. Install PyTorch (optional, for DQN agents):
```bash
pip install torch
```

## Quick Start

Run the interactive menu:

```bash
cd actions
python menu.py
```

This will present you with options to:
1. Run simulations
2. Conduct tournaments  
3. Analyze results
4. Train DQN models
5. Exit

## Game Rules

### Standard Oware Rules

- **Board**: 12 pits arranged in two rows (6 per player)
- **Setup**: 4 seeds per pit initially (48 total seeds)
- **Objective**: Capture more seeds than your opponent

### Gameplay

1. **Turn**: Players alternate turns, selecting one of their 6 pits
2. **Sowing**: Seeds from the selected pit are distributed counter-clockwise, one per pit
3. **Capturing**: If the last seed lands in an opponent's pit with 2 or 3 seeds total, capture all seeds in that pit
4. **Chain Captures**: Continue capturing backward if adjacent pits also have 2 or 3 seeds
5. **Winning**: First player to capture >24 seeds wins, or highest score when no moves remain

### Supported Variants

- **Standard**: 4 seeds per pit, chain captures enabled
- **Sparse**: 2 seeds per pit, chain captures enabled  
- **Dense**: 6 seeds per pit, chain captures enabled
- **No Chain**: 4 seeds per pit, only single pit captures

## AI Agents

### Basic Agents

| Agent | Strategy | Complexity |
|-------|----------|------------|
| **RandomAgent** | Random valid moves | Baseline |
| **GreedyAgent** | Maximizes immediate score gain | Simple |
| **HeuristicAgent** | Prefers captures, minimizes opponent opportunities | Moderate |

### Advanced Agents

| Agent | Algorithm | Features |
|-------|-----------|-----------|
| **MinimaxAgent** | Minimax with alpha-beta pruning | Configurable depth, position evaluation |
| **QLearningAgent** | Q-Learning | Exploration/exploitation, state-action values |

### Deep Learning Agents

| Agent | Architecture | Use Case |
|-------|-------------|----------|
| **DQNSmall** | Single hidden layer (64 units) | Fast training, lightweight |
| **DQNMedium** | Two hidden layers (128, 64) | Balanced performance |
| **DQNLarge** | Two hidden layers (256, 128) | Maximum performance |

All DQN agents feature:
- Experience replay buffer
- Target network for stable training
- ε-greedy exploration with decay
- Automatic fallback to NumPy implementation if PyTorch unavailable

## Usage

### Running Simulations

Simulate games between random AI matchups:

```bash
# From the actions directory
python simulation.py
```

Or use the menu system:
1. Select option `1: run simulation`
2. Enter number of episodes (e.g., 1000)
3. Results saved to `output/sim_log.csv`

### Tournament Mode

Run single-elimination tournaments:

```bash
# All available agents compete
python -c "import simulation; simulation.run_tournament()"
```

Or via menu option `2: run tournament`

### Training DQN Agents

Train deep reinforcement learning models:

1. Select menu option `4: train dqn`
2. Choose model size (Small/Medium/Large)
3. Select game variant
4. Set training episodes
5. Model checkpoints saved automatically

### Statistical Analysis

Analyze simulation results:

1. Run simulations first to generate data
2. Select menu option `3: run analysis`  
3. View detailed statistics in console
4. Results exported to `output/analysis_log.csv`

Analysis includes:
- Win rates by agent type
- Average game statistics
- Performance trends
- Head-to-head comparisons

## Project Structure

```
Oware-AI-Competition-Bot-Game/
├── owareEngine.py          # Core game engine and board logic
├── agents.py               # AI agent implementations
├── actions/                # Main application modules
│   ├── menu.py            # Interactive command-line interface
│   ├── simulation.py      # Game simulation and tournament logic  
│   └── analysis.py        # Statistical analysis and reporting
└── output/                # Generated logs and results
    ├── sim_log.csv        # Simulation results
    ├── tourney_log.csv    # Tournament results
    └── analysis_log.csv   # Statistical analysis output
```

### Key Components

- **`OwareBoard`**: Complete game state management with variant support
- **Agent Classes**: Modular AI implementations with consistent interface
- **DataLogger**: CSV logging for all game events and statistics
- **Training Pipeline**: Automated DQN training with progress tracking

## Configuration

### Game Variants

Modify game rules by specifying variants in `OwareBoard`:

```python
board = OwareBoard(variant="standard")  # Default
board = OwareBoard(variant="sparse")    # Fewer seeds
board = OwareBoard(variant="dense")     # More seeds  
board = OwareBoard(variant="no_chain")  # No chain captures
```

### Agent Parameters

Customize AI behavior:

```python
# Q-Learning agent with custom parameters
agent = QLearningAgent(
    learning_rate=0.1,
    discount_factor=0.95,
    exploration_rate=0.8
)

# DQN agent with custom architecture
agent = DQNAgent(
    hidden_sizes=(128, 64),
    lr=1e-3,
    buffer_size=10000
)
```

### Training Configuration

Adjust DQN training in the menu system or programmatically:

```python
agent = DQNMedium()
# Train against random opponents
for episode in range(1000):
    # ... training loop
    agent.train_step()
```

## Data Output

All game results are logged to CSV files in the `output/` directory:

### Simulation Logs (`sim_log.csv`)
- Episode number and agent types
- Winner and final scores  
- Game length and move counts
- Capture statistics
- Agent-specific metrics (e.g., epsilon values)

### Tournament Logs (`tourney_log.csv`)  
- Match identifiers and participants
- Best-of series results
- Aggregate match statistics
- Tournament progression data

### Analysis Logs (`analysis_log.csv`)
- Agent performance summaries
- Win rate calculations
- Statistical significance tests
- Performance trend analysis

## Contributing

Contributions welcome! Areas for enhancement:

1. **New AI Strategies**: Implement additional algorithms (MCTS, genetic algorithms, etc.)
2. **Advanced Analysis**: Add more sophisticated statistical methods
3. **GUI Interface**: Create graphical game visualization
4. **Network Play**: Add multiplayer capabilities
5. **Performance Optimization**: Improve simulation speed

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure existing tests pass
5. Submit a pull request

---

## License

This project is open source. Please see the repository for license details.

## Acknowledgments

- Traditional Oware game rules and variants
- PyTorch community for deep learning frameworks
- Reinforcement learning research community

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/jrain-dev/Oware-AI-Competition-Bot-Game).