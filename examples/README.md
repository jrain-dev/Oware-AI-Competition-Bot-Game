# Examples

This directory contains example scripts demonstrating usage of the Oware AI system.

## Available Examples

### run_with_stats.py

Demonstrates playing Oware matches with comprehensive statistics collection.

**Usage:**

```bash
# Run a single match with JSONL output
python3 examples/run_with_stats.py

# Run multiple matches with both JSONL and CSV output
python3 examples/run_with_stats.py --matches 10 --csv

# Specify custom output path
python3 examples/run_with_stats.py --output my_stats.jsonl --matches 5
```

**Options:**

- `--output PATH`: Output path for stats file (default: stats.jsonl)
- `--csv`: Also generate CSV output alongside JSONL
- `--matches N`: Number of matches to play (default: 1)

**Output:**

Creates statistics files containing detailed metrics about agent decision-making:
- Decision times (average, median, per-move)
- Match duration and move counts
- Final board states and scores
- Agent information and parameters

See [docs/metrics.md](../docs/metrics.md) for complete documentation on the metrics system.
