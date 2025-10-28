#!/usr/bin/env python3
"""
Example: Running an Oware match with statistics recording.

This script demonstrates how to use the StatsRecorder to collect detailed
metrics about agent decision-making during a match.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from owareEngine import OwareBoard
from agents import QLearningAgent, RandomAgent
from oware_ai.metrics import StatsRecorder
from oware_ai.metrics.stats import MatchRecorder


def play_match_with_stats(agent0, agent1, match_id="example_match", 
                          stats_output="stats.jsonl", enable_csv=True):
    """
    Play a single match between two agents and record statistics.
    
    Args:
        agent0: First agent (player 0)
        agent1: Second agent (player 1)
        match_id: Unique identifier for this match
        stats_output: Path to output JSONL file
        enable_csv: Whether to also write CSV format
    
    Returns:
        Winner ID (0, 1, or None for draw)
    """
    # Initialize stats recorder
    stats_recorder = StatsRecorder(stats_output, enable_csv=enable_csv)
    
    if not stats_recorder.enabled:
        print("Warning: Stats recording is disabled due to file errors")
    
    # Prepare agent info (only include serializable params)
    def get_serializable_params(agent):
        """Extract serializable parameters from agent."""
        params = {}
        agent_dict = getattr(agent, '__dict__', {})
        for key, value in agent_dict.items():
            # Only include simple types that are JSON serializable
            if isinstance(value, (int, float, str, bool, type(None))):
                params[key] = value
        return params
    
    agent0_info = {
        "name": "Player0",
        "type": type(agent0).__name__,
        "params": get_serializable_params(agent0)
    }
    
    agent1_info = {
        "name": "Player1",
        "type": type(agent1).__name__,
        "params": get_serializable_params(agent1)
    }
    
    # Create match recorder
    match_recorder = MatchRecorder(
        match_id=match_id,
        agent0_info=agent0_info,
        agent1_info=agent1_info,
        seed=None
    )
    
    # Initialize board and agents
    board = OwareBoard()
    agents = {0: agent0, 1: agent1}
    state = board.reset()
    done = False
    move_number = 0
    
    print(f"\n=== Starting match: {match_id} ===")
    print(f"Player 0: {agent0_info['type']}")
    print(f"Player 1: {agent1_info['type']}")
    print(f"Stats output: {stats_output}")
    print()
    
    # Reset agent state if needed
    if hasattr(agent0, "end_episode"):
        agent0.end_episode()
    if hasattr(agent1, "end_episode"):
        agent1.end_episode()
    
    # Game loop with timing
    while not done:
        current_player_id = board.current_player
        current_agent = agents[current_player_id]
        valid_moves = board.get_valid_moves(current_player_id)
        
        if not valid_moves:
            break
        
        # Time the agent's decision
        with match_recorder.time_move(move_number, current_player_id) as timer:
            action = current_agent.select_action(state, valid_moves)
            
            # Try to extract nodes expanded if available
            if hasattr(current_agent, 'nodes_expanded'):
                timer.set_nodes_expanded(current_agent.nodes_expanded)
            
            # Try to extract evaluation score if available
            if hasattr(current_agent, 'last_evaluation'):
                timer.set_evaluation_score(current_agent.last_evaluation)
        
        # Record the move metrics
        metrics = timer.get_metrics(
            move_number=move_number,
            player=current_player_id,
            move_selected=action,
            valid_moves=valid_moves
        )
        match_recorder.record_move(metrics)
        
        # Apply the move
        reward, next_state, done = board.apply_move(action)
        
        # Update learning agent if applicable
        if current_player_id == 0 and hasattr(agent0, "update"):
            next_valid_moves = board.get_valid_moves(board.current_player)
            agent0.update(reward, next_state, next_valid_moves)
        
        state = next_state
        move_number += 1
    
    # Finalize match metrics
    final_metrics = match_recorder.finalize(
        board_state=board.board.tolist(),
        scores=board.scores,
        winner=board.winner
    )
    
    # Record to stats file
    if stats_recorder.enabled:
        stats_recorder.record_match(final_metrics)
        print(f"✓ Match statistics recorded to {stats_output}")
    
    # Print summary
    print(f"\nMatch completed:")
    print(f"  Moves: {final_metrics.moves_count}")
    print(f"  Duration: {final_metrics.duration_seconds:.3f}s")
    print(f"  Avg decision time: {final_metrics.avg_decision_time*1000:.2f}ms")
    print(f"  Winner: Player {board.winner if board.winner is not None else 'Draw'}")
    print(f"  Final scores: P0={board.scores[0]}, P1={board.scores[1]}")
    
    if final_metrics.total_nodes_expanded is not None:
        print(f"  Total nodes expanded: {final_metrics.total_nodes_expanded}")
        print(f"  Avg nodes per move: {final_metrics.avg_nodes_expanded_per_move:.1f}")
    
    return board.winner


def main():
    """Run example matches with stats recording."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Play Oware match with stats recording")
    parser.add_argument("--output", default="stats.jsonl", 
                       help="Output path for stats (default: stats.jsonl)")
    parser.add_argument("--csv", action="store_true", 
                       help="Also output CSV format")
    parser.add_argument("--matches", type=int, default=1,
                       help="Number of matches to play (default: 1)")
    
    args = parser.parse_args()
    
    # Play multiple matches
    for i in range(args.matches):
        # Create fresh agents for each match
        agent0 = QLearningAgent(exploration_rate=0.3)
        agent1 = RandomAgent()
        
        match_id = f"match_{i+1:03d}"
        play_match_with_stats(
            agent0, 
            agent1, 
            match_id=match_id,
            stats_output=args.output,
            enable_csv=args.csv
        )
        print()
    
    print(f"\n{'='*50}")
    print(f"All matches completed!")
    print(f"Statistics saved to: {args.output}")
    if args.csv:
        csv_path = Path(args.output).with_suffix('.csv')
        print(f"CSV statistics saved to: {csv_path}")


if __name__ == "__main__":
    main()
