# Tests

This directory contains unit and integration tests for the Oware AI project.

## Running Tests

### Unit Tests

Test the metrics collection components:

```bash
PYTHONPATH=. python3 tests/test_stats_recorder.py
```

### Integration Tests

Test the complete system integration:

```bash
python3 tests/test_integration.py
```

### All Tests

Run all tests:

```bash
PYTHONPATH=. python3 tests/test_stats_recorder.py && python3 tests/test_integration.py
```

## Test Coverage

- `test_stats_recorder.py`: Unit tests for StatsRecorder, DecisionTimer, MatchRecorder
- `test_integration.py`: End-to-end integration tests for the metrics system

## Expected Output

All tests should pass with output indicating successful test completion:

```
✓ test_decision_timer passed
✓ test_decision_timer_with_nodes passed
✓ test_stats_recorder_jsonl passed
✓ test_stats_recorder_csv passed
✓ test_stats_recorder_invalid_path passed
✓ test_match_recorder passed

All tests passed!
```
