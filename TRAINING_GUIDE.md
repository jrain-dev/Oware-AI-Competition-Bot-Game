# Deep Learning Training System Documentation

## Overview

The Oware AI project now includes a comprehensive deep learning training system with advanced features for training, monitoring, and managing DQN (Deep Q-Network) models.

## Features

### 🚀 Comprehensive Training Pipeline
- **Configurable Parameters**: Customize training episodes, evaluation intervals, checkpoint frequency
- **Multi-Opponent Curriculum**: Train against different opponent types with adaptive difficulty
- **Progress Tracking**: Real-time monitoring of win rates, losses, and exploration rates
- **Early Stopping**: Automatic termination when no improvement is detected

### 📊 Advanced Metrics & Logging
- **Training Metrics**: Loss tracking, win rate evolution, episode statistics
- **Evaluation System**: Regular testing against all opponent types
- **Detailed Logging**: JSON logs with comprehensive training data
- **Performance Monitoring**: Track improvement over time

### 💾 Sophisticated Checkpoint System
- **Automatic Saving**: Regular checkpoint creation during training
- **Best Model Tracking**: Automatically saves the best-performing model
- **Versioning**: Multiple checkpoint versions with metadata
- **Easy Loading**: Simple model restoration from checkpoints

### 🎮 User-Friendly Interface
- **Menu Integration**: Easy access through the main menu system
- **Quick Training**: Fast setup with default parameters
- **Advanced Configuration**: Full control over training parameters
- **Model Management**: Browse, evaluate, and compare trained models

## Quick Start

### 1. Basic Training (Menu)
```bash
cd actions
python3 menu.py
# Select option 4 (train dqn) -> option 1 (Quick Training)
```

### 2. Command Line Training
```bash
cd actions
python3 training.py small 2000    # Train DQNSmall for 2000 episodes
python3 training.py medium 5000   # Train DQNMedium for 5000 episodes
python3 training.py large 10000   # Train DQNLarge for 10000 episodes
```

### 3. Evaluate Trained Models
```bash
python3 training.py evaluate path/to/checkpoint.pth small 100
```

## Training Configuration

### TrainingConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_episodes` | 5000 | Total training episodes |
| `warmup_episodes` | 100 | Episodes before training starts |
| `eval_interval` | 250 | Episodes between evaluations |
| `eval_episodes` | 50 | Episodes per evaluation |
| `checkpoint_interval` | 500 | Episodes between checkpoints |
| `patience` | 1000 | Early stopping patience |
| `variant` | 'standard' | Game variant to train on |

### Advanced Configuration Example
```python
from actions.training import create_training_config, DQNTrainer
from agents import DQNMedium

config = create_training_config(
    total_episodes=10000,
    eval_interval=200,
    checkpoint_interval=400,
    patience=1500,
    variant='dense',
    opponent_weights=[0.3, 0.3, 0.3, 0.1]  # More challenging opponents
)

trainer = DQNTrainer(config)
results = trainer.train(DQNMedium, 'my_advanced_training')
```

## Model Architecture

### Available Model Sizes

| Model | Hidden Layers | Parameters | Use Case |
|-------|---------------|------------|----------|
| **DQNSmall** | (64,) | ~1K | Fast training, prototyping |
| **DQNMedium** | (128, 64) | ~11K | Balanced performance |
| **DQNLarge** | (256, 128) | ~42K | Maximum performance |

### Training Features per Model
- **Experience Replay Buffer**: Stores recent experiences for stable learning
- **Target Network**: Separate network for stable Q-value targets
- **ε-greedy Exploration**: Decreasing exploration rate over time
- **Automatic Normalization**: Input state normalization for stable training

## File Structure

```
output/training/
├── [session_name]/
│   ├── config.json              # Training configuration
│   ├── training_log.json        # Episode-by-episode logs
│   ├── final_metrics.pkl        # Complete training metrics
│   └── checkpoints/
│       ├── checkpoint_history.json     # Checkpoint metadata
│       ├── best_checkpoint.pth         # Best model (symlink)
│       ├── checkpoint_ep500_*.pth      # Regular checkpoints
│       ├── checkpoint_ep500_*_metadata.json
│       └── checkpoint_ep500_*_metrics.pkl
```

## Training Opponents

The system trains against a curriculum of opponents:

| Opponent | Strategy | Default Weight |
|----------|----------|----------------|
| **Random** | Random moves | 40% |
| **Greedy** | Maximize immediate score | 30% |
| **Heuristic** | Capture preference + blocking | 20% |
| **Minimax** | Minimax with alpha-beta pruning | 10% |

## Performance Monitoring

### Real-time Metrics
- **Win Rate**: Recent performance against opponents
- **Exploration Rate (ε)**: Current exploration probability  
- **Training Loss**: Neural network loss values
- **Episodes/Second**: Training speed

### Evaluation Reports
```
Evaluation results:
  random_win_rate: 0.820
  greedy_win_rate: 0.640
  heuristic_win_rate: 0.580
  minimax_win_rate: 0.420
  overall_win_rate: 0.615
```

## Best Practices

### 1. Training Configuration
- **Start Small**: Begin with DQNSmall and fewer episodes
- **Gradual Increase**: Scale up model size and episodes based on results
- **Monitor Early**: Use frequent evaluation intervals initially
- **Patience Setting**: Set patience to ~20% of total episodes

### 2. Performance Optimization
- **GPU Acceleration**: Training automatically uses GPU when available
- **Batch Size**: Larger models can handle larger batch sizes
- **Buffer Size**: Increase for longer training sessions

### 3. Model Selection
- **Evaluation**: Always evaluate models against multiple opponents
- **Consistency**: Look for stable performance across opponent types
- **Generalization**: Test on different game variants

## Troubleshooting

### Common Issues

1. **PyTorch Not Available**
   ```bash
   pip install torch
   ```

2. **CUDA Out of Memory**
   - Reduce batch size in model configuration
   - Use smaller model variant

3. **Training Stagnation**
   - Increase exploration rate
   - Adjust opponent curriculum
   - Try different game variants

4. **Checkpoint Loading Errors**
   - Check file permissions
   - Verify model type matches checkpoint
   - Ensure PyTorch version compatibility

### Debug Mode
Enable detailed logging by setting `detailed_logging=True` in training configuration.

## Advanced Features

### Custom Opponent Training
```python
# Create custom opponent mix
config = create_training_config(
    opponent_types=['random', 'greedy', 'minimax'],
    opponent_weights=[0.5, 0.3, 0.2]
)
```

### Multi-Variant Training
```python
# Train on different game variants
variants = ['standard', 'sparse', 'dense', 'no_chain']
for variant in variants:
    config = create_training_config(variant=variant)
    trainer = DQNTrainer(config)
    trainer.train(DQNMedium, f'medium_{variant}')
```

### Hyperparameter Sweeps
```python
# Systematic hyperparameter exploration
learning_rates = [1e-3, 5e-4, 1e-4]
for lr in learning_rates:
    agent = DQNMedium()
    agent.opt = optim.Adam(agent.net.parameters(), lr=lr)
    # Train with modified agent...
```

## Integration with Existing System

The training system seamlessly integrates with the existing Oware AI framework:

- **Compatible Agents**: Works with all existing DQN variants
- **Game Engine**: Uses the same OwareBoard for consistency
- **Logging**: Integrates with existing CSV logging system
- **Menu System**: Accessible through the main menu interface

## Future Enhancements

Potential improvements include:
- **Resume Training**: Continue from previous checkpoints
- **Model Comparison**: Side-by-side performance analysis
- **Distributed Training**: Multi-process training support
- **Visualization**: Training progress plots and charts
- **Tournament Integration**: Automatic tournament participation for trained models