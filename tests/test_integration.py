"""Integration tests for the metrics collection system."""

import os
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulationRunner import run_simulation
from owareEngine import OwareBoard
from agents import QLearningAgent, RandomAgent
from oware_ai.metrics import StatsRecorder
from oware_ai.metrics.stats import MatchRecorder


def test_simulation_with_stats():
    """Test that run_simulation works with stats enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "test_stats.jsonl"
        
        run_simulation(episodes=3, stats_output=str(stats_path), enable_stats_csv=False)
        
        # Verify stats file was created
        assert stats_path.exists(), "Stats file not created"
        
        # Verify contents
        with open(stats_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"
            
            # Parse first line
            data = json.loads(lines[0])
            assert 'match_id' in data
            assert 'timestamp' in data
            assert 'duration_seconds' in data
            assert 'moves_count' in data
            assert data['moves_count'] > 0


def test_simulation_without_stats():
    """Test that run_simulation works with stats disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.getcwd()
        try:
            os.chdir(tmpdir)
            run_simulation(episodes=2, stats_output="", enable_stats_csv=False)
            
            # Verify no stats file was created
            stats_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(stats_files) == 0, f"Stats file created when it shouldn't be: {stats_files}"
        finally:
            os.chdir(orig_dir)


def test_manual_match_with_stats():
    """Test using StatsRecorder and MatchRecorder directly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "manual_stats.jsonl"
        
        # Create a quick match
        stats_recorder = StatsRecorder(str(stats_path), enable_csv=False)
        match_recorder = MatchRecorder(
            match_id="integration_test",
            agent0_info={"name": "A0", "type": "Q", "params": {}},
            agent1_info={"name": "A1", "type": "R", "params": {}},
        )
        
        board = OwareBoard()
        agent0 = QLearningAgent()
        agent1 = RandomAgent()
        agents = {0: agent0, 1: agent1}
        
        state = board.reset()
        move_num = 0
        
        while not board.game_over and move_num < 100:
            player = board.current_player
            agent = agents[player]
            valid_moves = board.get_valid_moves(player)
            
            if not valid_moves:
                break
            
            with match_recorder.time_move(move_num, player) as timer:
                move = agent.select_action(state, valid_moves)
            
            metrics = timer.get_metrics(move_num, player, move, valid_moves)
            match_recorder.record_move(metrics)
            
            reward, state, done = board.apply_move(move)
            move_num += 1
            
            if done:
                break
        
        final_metrics = match_recorder.finalize(
            board_state=board.board.tolist(),
            scores=board.scores,
            winner=board.winner
        )
        
        stats_recorder.record_match(final_metrics)
        
        # Verify
        assert stats_path.exists()
        with open(stats_path, 'r') as f:
            data = json.loads(f.readline())
            assert data['match_id'] == "integration_test"
            assert data['moves_count'] == move_num
            assert data['moves_count'] > 0


def test_env_var_stats_output():
    """Test that STATS_OUTPUT environment variable works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stats_path = Path(tmpdir) / "env_stats.jsonl"
        
        # Set environment variable
        old_val = os.environ.get('STATS_OUTPUT')
        os.environ['STATS_OUTPUT'] = str(stats_path)
        
        try:
            # Run with default (should use env var)
            run_simulation(episodes=2, stats_output=None, enable_stats_csv=False)
            
            # Verify stats file was created at the env var location
            assert stats_path.exists(), f"Stats file not created at {stats_path}"
            
            with open(stats_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
        finally:
            # Restore original env var
            if old_val is None:
                os.environ.pop('STATS_OUTPUT', None)
            else:
                os.environ['STATS_OUTPUT'] = old_val


if __name__ == "__main__":
    test_simulation_with_stats()
    print("✓ test_simulation_with_stats passed")
    
    test_simulation_without_stats()
    print("✓ test_simulation_without_stats passed")
    
    test_manual_match_with_stats()
    print("✓ test_manual_match_with_stats passed")
    
    test_env_var_stats_output()
    print("✓ test_env_var_stats_output passed")
    
    print("\nAll integration tests passed!")
