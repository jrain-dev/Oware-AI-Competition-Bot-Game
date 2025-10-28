# Metrics Collection Documentation

## Overview

The Oware AI repository now includes comprehensive metrics collection for AI agent decision-making. This system tracks timing, nodes expanded, evaluation scores, and other statistics for every match played.

## Features

- **Automatic Metrics Collection**: Minimal code changes required to enable stats recording
- **Multiple Output Formats**: JSONL (default) and CSV
- **Thread-Safe**: Safe for concurrent recording
- **Configurable**: Enable/disable via environment variables or function parameters
- **Rich Metrics**: Captures decision times, node expansions, evaluations, and more

## Quick Start

### Basic Usage with Example Script

Run a match with stats recording:

```bash
python3 examples/run_with_stats.py --matches 3 --csv
```

This will create:
- `stats.jsonl` - Detailed match metrics in JSONL format
- `stats.csv` - Summary metrics in CSV format

### Using Environment Variables

Enable stats recording via environment variable:

```bash
STATS_OUTPUT="my_stats.jsonl" python3 -c "from simulationRunner import run_simulation; run_simulation(episodes=100)"
```

Disable stats recording:

```bash
STATS_OUTPUT="" python3 -c "from simulationRunner import run_simulation; run_simulation(episodes=100)"
```

### Programmatic Usage

```python
from simulationRunner import run_simulation

# With stats recording (default: stats.jsonl)
run_simulation(episodes=1000, stats_output="training_stats.jsonl", enable_stats_csv=True)

# Without stats recording
run_simulation(episodes=1000, stats_output="")
```

## Output Format

### JSONL Format

Each line in the JSONL file is a complete JSON object for one match:

```json
{
  "match_id": "episode_000001",
  "timestamp": "2025-10-28T12:34:56.789012",
  "duration_seconds": 2.5,
  "moves_count": 85,
  "avg_decision_time": 0.000012,
  "median_decision_time": 0.000010,
  "decision_times": [0.000011, 0.000012, ...],
  "total_nodes_expanded": null,
  "avg_nodes_expanded_per_move": null,
  "evaluations": [null, null, ...],
  "final_board_state": [0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0],
  "final_scores": [26, 22],
  "winner": 0,
  "agent0_info": {
    "name": "Player0",
    "type": "QLearningAgent",
    "params": {"lr": 0.1, "gamma": 0.95, "epsilon": 0.999}
  },
  "agent1_info": {
    "name": "Player1",
    "type": "RandomAgent",
    "params": {}
  },
  "seed": null,
  "metadata": {}
}
```

### CSV Format

Simplified summary format suitable for spreadsheet analysis:

| match_id | timestamp | duration_seconds | moves_count | avg_decision_time | winner | agent0_type | agent1_type |
|----------|-----------|------------------|-------------|-------------------|--------|-------------|-------------|
| episode_000001 | 2025-10-28... | 2.5000 | 85 | 0.000012 | 0 | QLearningAgent | RandomAgent |

## Available Metrics

### Per-Match Metrics

- `match_id`: Unique identifier for the match
- `timestamp`: ISO 8601 timestamp of when match completed
- `duration_seconds`: Total match duration in seconds
- `moves_count`: Total number of moves in the match
- `avg_decision_time`: Average time per decision (seconds)
- `median_decision_time`: Median decision time (seconds)
- `decision_times`: List of all decision times
- `total_nodes_expanded`: Total nodes expanded (if available)
- `avg_nodes_expanded_per_move`: Average nodes per move (if available)
- `evaluations`: List of evaluation scores (if available)
- `final_board_state`: Final pit values [12 integers]
- `final_scores`: Final scores [player0_score, player1_score]
- `winner`: Winner ID (0, 1, or null for draw)
- `agent0_info`: Agent 0 information (name, type, params)
- `agent1_info`: Agent 1 information
- `seed`: Random seed used (if applicable)
- `metadata`: Additional custom metadata

### Per-Move Metrics (in decision_times list)

Each move records:
- Decision time (wall-clock, high-resolution)
- Nodes expanded (if agent exposes this)
- Evaluation score (if agent returns this)
- Move selected
- Valid moves available

## Integration into Your Code

### Minimal Integration Example

```python
from oware_ai.metrics import StatsRecorder
from oware_ai.metrics.stats import MatchRecorder

# Initialize stats recorder
stats_recorder = StatsRecorder("output.jsonl", enable_csv=True)

# For each match
match_recorder = MatchRecorder(
    match_id="match_001",
    agent0_info={"name": "A0", "type": "QLearning", "params": {}},
    agent1_info={"name": "A1", "type": "Random", "params": {}},
)

# During game loop
for move_num, (player, agent) in enumerate(game_loop):
    with match_recorder.time_move(move_num, player) as timer:
        move = agent.select_action(state, valid_moves)
        # Optionally set additional metrics
        if hasattr(agent, 'nodes_expanded'):
            timer.set_nodes_expanded(agent.nodes_expanded)
    
    metrics = timer.get_metrics(move_num, player, move, valid_moves)
    match_recorder.record_move(metrics)

# After match completes
final_metrics = match_recorder.finalize(
    board_state=board.board.tolist(),
    scores=board.scores,
    winner=board.winner
)
stats_recorder.record_match(final_metrics)
```

### Extending Agent Classes

To expose node expansion or evaluation scores, add attributes to your agent:

```python
class MyAgent(Agent):
    def select_action(self, state, valid_moves):
        self.nodes_expanded = 0
        self.last_evaluation = None
        
        # ... decision logic ...
        self.nodes_expanded = 150  # Track nodes
        self.last_evaluation = 0.75  # Track evaluation
        
        return best_move
```

The metrics system will automatically pick up these attributes if present.

## Configuration Options

### StatsRecorder Parameters

- `output_path` (str): Path to JSONL file (default: "stats.jsonl")
- `enable_csv` (bool): Also write CSV format (default: False)
- `warn_on_error` (bool): Print warnings on file errors (default: True)

### Environment Variables

- `STATS_OUTPUT`: Override default stats output path
  - Set to empty string `""` to disable stats recording
  - Default: `"stats.jsonl"`

## Error Handling

The stats recorder fails gracefully:
- If output path is not writable, it prints a warning and disables recording
- Game execution continues normally even if stats recording fails
- No exceptions are raised to calling code

## Performance Impact

The metrics collection system is designed to have minimal performance impact:
- Uses high-resolution timers (`time.perf_counter`)
- Thread-safe append-only writes
- No buffering overhead (writes complete records)
- Typical overhead: < 1% for decision times > 1ms

## Testing

Run the test suite:

```bash
PYTHONPATH=. python3 tests/test_stats_recorder.py
```

All tests should pass with output:
```
✓ test_decision_timer passed
✓ test_decision_timer_with_nodes passed
✓ test_stats_recorder_jsonl passed
✓ test_stats_recorder_csv passed
✓ test_stats_recorder_invalid_path passed
✓ test_match_recorder passed

All tests passed!
```

## Analysis Examples

### Loading and Analyzing JSONL Data

```python
import json
import pandas as pd

# Load all matches
matches = []
with open('stats.jsonl', 'r') as f:
    for line in f:
        matches.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(matches)

# Analyze average decision times by agent type
print(df.groupby('agent0_info')['avg_decision_time'].mean())

# Find longest matches
print(df.nlargest(10, 'moves_count'))
```

### Loading CSV Data

```python
import pandas as pd

df = pd.read_csv('stats.csv')
print(df.describe())
```

## Backwards Compatibility

- Stats recording is opt-in via parameters or environment variables
- Default behavior when not configured: writes to `./stats.jsonl`
- Existing code continues to work without modification
- No changes to game logic or agent behavior

## Future Enhancements

Possible future additions:
- SQLite output format
- Real-time dashboard/visualization
- Memory usage tracking
- Per-agent comparison reports
- Automated performance regression detection
