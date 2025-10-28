"""Tests for the StatsRecorder and related metrics utilities."""

import json
import tempfile
from pathlib import Path

from oware_ai.metrics import StatsRecorder, DecisionTimer, MoveMetrics
from oware_ai.metrics.stats import MatchRecorder, MatchMetrics


def test_decision_timer():
    """Test DecisionTimer context manager."""
    import time
    
    with DecisionTimer() as timer:
        time.sleep(0.01)  # Sleep for 10ms
    
    assert timer.elapsed >= 0.01
    assert timer.elapsed < 0.05  # Should be well under 50ms
    
    # Test metrics creation
    metrics = timer.get_metrics(move_number=1, player=0, move_selected=3, valid_moves=[0, 1, 2, 3])
    assert metrics.move_number == 1
    assert metrics.player == 0
    assert metrics.decision_time >= 0.01
    assert metrics.move_selected == 3
    assert metrics.valid_moves == [0, 1, 2, 3]


def test_decision_timer_with_nodes():
    """Test DecisionTimer with nodes expanded."""
    with DecisionTimer() as timer:
        timer.set_nodes_expanded(100)
        timer.set_evaluation_score(0.75)
    
    metrics = timer.get_metrics(move_number=2, player=1)
    assert metrics.nodes_expanded == 100
    assert metrics.evaluation_score == 0.75


def test_stats_recorder_jsonl():
    """Test StatsRecorder writing to JSONL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_stats.jsonl"
        recorder = StatsRecorder(str(output_path), enable_csv=False)
        
        assert recorder.enabled
        
        # Create sample metrics
        metrics = MatchMetrics(
            match_id="test_001",
            timestamp="2025-01-01T00:00:00",
            duration_seconds=10.5,
            moves_count=20,
            avg_decision_time=0.5,
            median_decision_time=0.4,
            decision_times=[0.3, 0.4, 0.5, 0.6, 0.7],
            total_nodes_expanded=500,
            avg_nodes_expanded_per_move=25.0,
            evaluations=[0.1, 0.2, 0.3],
            final_board_state=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            final_scores=[25, 23],
            winner=0,
            agent0_info={"name": "Agent0", "type": "QLearningAgent"},
            agent1_info={"name": "Agent1", "type": "RandomAgent"},
            seed=42
        )
        
        # Record the match
        success = recorder.record_match(metrics)
        assert success
        
        # Verify JSONL was written
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            
            data = json.loads(lines[0])
            assert data['match_id'] == "test_001"
            assert data['duration_seconds'] == 10.5
            assert data['moves_count'] == 20
            assert data['winner'] == 0
            assert data['agent0_info']['type'] == "QLearningAgent"


def test_stats_recorder_csv():
    """Test StatsRecorder writing to both JSONL and CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_stats.jsonl"
        recorder = StatsRecorder(str(output_path), enable_csv=True)
        
        assert recorder.enabled
        assert recorder.csv_path is not None
        
        metrics = MatchMetrics(
            match_id="test_002",
            timestamp="2025-01-01T00:00:00",
            duration_seconds=5.0,
            moves_count=10,
            avg_decision_time=0.25,
            median_decision_time=0.20,
            decision_times=[0.2, 0.3],
            total_nodes_expanded=None,
            avg_nodes_expanded_per_move=None,
            evaluations=[],
            final_board_state=[0] * 12,
            final_scores=[24, 24],
            winner=None,
            agent0_info={"name": "A0", "type": "Random"},
            agent1_info={"name": "A1", "type": "Random"},
            seed=None
        )
        
        recorder.record_match(metrics)
        
        # Verify CSV exists and has content
        assert recorder.csv_path.exists()
        
        with open(recorder.csv_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Header + 1 data row
            assert 'match_id' in lines[0]
            assert 'test_002' in lines[1]


def test_stats_recorder_invalid_path():
    """Test StatsRecorder with invalid output path."""
    recorder = StatsRecorder("/nonexistent/path/stats.jsonl", warn_on_error=False)
    assert not recorder.enabled


def test_match_recorder():
    """Test MatchRecorder helper class."""
    agent0_info = {"name": "Agent0", "type": "QLearning"}
    agent1_info = {"name": "Agent1", "type": "Random"}
    
    match_rec = MatchRecorder("match_123", agent0_info, agent1_info, seed=42)
    
    # Simulate recording some moves
    with match_rec.time_move(0, 0) as timer:
        # Simulate agent thinking
        import time
        time.sleep(0.01)
        timer.set_nodes_expanded(50)
    
    metrics1 = timer.get_metrics(move_number=0, player=0, move_selected=2, valid_moves=[0, 1, 2, 3])
    match_rec.record_move(metrics1)
    
    with match_rec.time_move(1, 1) as timer:
        time.sleep(0.01)
        timer.set_nodes_expanded(30)
    
    metrics2 = timer.get_metrics(move_number=1, player=1, move_selected=8, valid_moves=[6, 7, 8, 9])
    match_rec.record_move(metrics2)
    
    # Finalize match
    final_metrics = match_rec.finalize(
        board_state=[0] * 12,
        scores=[26, 22],
        winner=0
    )
    
    assert final_metrics.match_id == "match_123"
    assert final_metrics.moves_count == 2
    assert final_metrics.winner == 0
    assert final_metrics.total_nodes_expanded == 80
    assert final_metrics.seed == 42


if __name__ == "__main__":
    # Run tests manually
    test_decision_timer()
    print("✓ test_decision_timer passed")
    
    test_decision_timer_with_nodes()
    print("✓ test_decision_timer_with_nodes passed")
    
    test_stats_recorder_jsonl()
    print("✓ test_stats_recorder_jsonl passed")
    
    test_stats_recorder_csv()
    print("✓ test_stats_recorder_csv passed")
    
    test_stats_recorder_invalid_path()
    print("✓ test_stats_recorder_invalid_path passed")
    
    test_match_recorder()
    print("✓ test_match_recorder passed")
    
    print("\nAll tests passed!")
