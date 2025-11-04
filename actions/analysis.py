import csv
import os

class DataLogger:
    def __init__(self, filename):
        # Always save to output directory
        if not filename.startswith('output/'):
            # Get the directory of this file (actions/) and go up one level to the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, 'output')
            self.filename = os.path.join(output_dir, filename)
        else:
            self.filename = filename
            
        self.data = []
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        # Initialize CSV file with headers
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Initialize CSV file with headers, overwriting any existing file"""
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'episode', 'agent0_type', 'agent1_type', 'winner', 
                'agent0_score', 'agent1_score', 'total_moves', 'game_length',
                'agent0_captures', 'agent1_captures', 'final_board_seeds',
                'score_difference', 'winner_margin', 'agent0_epsilon', 'agent1_epsilon'
            ])
    
    def log_episode(self, episode_data):
        """Log episode data to both memory and CSV file"""
        self.data.append(episode_data)
        
        # Write to CSV immediately
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [
                episode_data.get('episode', ''),
                episode_data.get('agent0_type', ''),
                episode_data.get('agent1_type', ''),
                episode_data.get('winner', ''),
                episode_data.get('agent0_score', ''),
                episode_data.get('agent1_score', ''),
                episode_data.get('total_moves', ''),
                episode_data.get('game_length', ''),
                episode_data.get('agent0_captures', ''),
                episode_data.get('agent1_captures', ''),
                episode_data.get('final_board_seeds', ''),
                episode_data.get('score_difference', ''),
                episode_data.get('winner_margin', ''),
                episode_data.get('agent0_epsilon', ''),
                episode_data.get('agent1_epsilon', '')
            ]
            writer.writerow(row)
    
    def get_data(self):
        """Return all logged data"""
        return self.data
    
    def save(self, filename=None):
        """Save data to CSV file (already done incrementally)"""
        if filename:
            self.filename = filename
            self._save_all_data()
    
    def _save_all_data(self):
        """Save all data to CSV file (used when filename changes)"""
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'agent0_type', 'agent1_type', 'winner', 'scores', 'moves'])
            for episode_data in self.data:
                row = [
                    episode_data.get('episode', ''),
                    episode_data.get('agent0_type', ''),
                    episode_data.get('agent1_type', ''),
                    episode_data.get('winner', ''),
                    episode_data.get('scores', ''),
                    episode_data.get('moves', '')
                ]
                writer.writerow(row)


def analyze_sim_data(sim_log_path):
    """Perform deep statistical analysis on simulation data"""
    import pandas as pd
    import numpy as np
    from collections import defaultdict
    import statistics
    
    # Read the simulation data
    try:
        df = pd.read_csv(sim_log_path)
    except FileNotFoundError:
        print(f"Error: {sim_log_path} not found. Please run simulations first.")
        return None
    except Exception as e:
        print(f"Error reading {sim_log_path}: {e}")
        return None
    
    if df.empty:
        print("No data found in simulation log.")
        return None
    
    # Get all unique agent types
    agent_types = set()
    agent_types.update(df['agent0_type'].unique())
    agent_types.update(df['agent1_type'].unique())
    agent_types.discard('N/A')  # Remove any N/A entries
    
    analysis_results = []
    
    for agent_type in agent_types:
        # Get all games where this agent participated
        agent_games = df[(df['agent0_type'] == agent_type) | (df['agent1_type'] == agent_type)].copy()
        
        if agent_games.empty:
            continue
            
        # Initialize statistics
        stats = {
            'agent_type': agent_type,
            'total_games': len(agent_games),
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'total_score': 0,
            'opponent_score': 0,
            'total_moves': 0,
            'game_lengths': [],
            'win_margins': [],
            'loss_margins': [],
            'captures_per_game': [],
            'scores_when_agent0': [],
            'scores_when_agent1': [],
            'epsilon_values': []
        }
        
        # Analyze each game
        for _, game in agent_games.iterrows():
            is_agent0 = game['agent0_type'] == agent_type
            agent_score = game['agent0_score'] if is_agent0 else game['agent1_score']
            opponent_score = game['agent1_score'] if is_agent0 else game['agent0_score']
            
            # Convert to numeric, handling any string values
            try:
                agent_score = float(agent_score) if pd.notna(agent_score) else 0
                opponent_score = float(opponent_score) if pd.notna(opponent_score) else 0
                game_length = float(game['game_length']) if pd.notna(game['game_length']) else 0
                total_moves = float(game['total_moves']) if pd.notna(game['total_moves']) else 0
            except (ValueError, TypeError):
                continue
                
            stats['total_score'] += agent_score
            stats['opponent_score'] += opponent_score
            stats['total_moves'] += total_moves
            stats['game_lengths'].append(game_length)
            stats['captures_per_game'].append(agent_score)
            
            if is_agent0:
                stats['scores_when_agent0'].append(agent_score)
                epsilon_val = game.get('agent0_epsilon', 'N/A')
            else:
                stats['scores_when_agent1'].append(agent_score)
                epsilon_val = game.get('agent1_epsilon', 'N/A')
                
            # Track epsilon values if available
            if epsilon_val != 'N/A':
                try:
                    stats['epsilon_values'].append(float(epsilon_val))
                except (ValueError, TypeError):
                    pass
            
            # Determine win/loss/draw
            winner = game['winner']
            if winner == agent_type:
                stats['wins'] += 1
                margin = agent_score - opponent_score
                stats['win_margins'].append(margin)
            elif winner == 'Draw':
                stats['draws'] += 1
            else:
                stats['losses'] += 1
                margin = opponent_score - agent_score
                stats['loss_margins'].append(margin)
        
        # Calculate derived statistics
        if stats['total_games'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_games']
            stats['loss_rate'] = stats['losses'] / stats['total_games']
            stats['draw_rate'] = stats['draws'] / stats['total_games']
            stats['avg_score_per_game'] = stats['total_score'] / stats['total_games']
            stats['avg_opponent_score'] = stats['opponent_score'] / stats['total_games']
            stats['avg_score_differential'] = (stats['total_score'] - stats['opponent_score']) / stats['total_games']
            stats['avg_game_length'] = statistics.mean(stats['game_lengths']) if stats['game_lengths'] else 0
            stats['avg_total_moves'] = stats['total_moves'] / stats['total_games']
            
            # Win/loss margin statistics
            stats['avg_win_margin'] = statistics.mean(stats['win_margins']) if stats['win_margins'] else 0
            stats['avg_loss_margin'] = statistics.mean(stats['loss_margins']) if stats['loss_margins'] else 0
            stats['max_win_margin'] = max(stats['win_margins']) if stats['win_margins'] else 0
            stats['max_loss_margin'] = max(stats['loss_margins']) if stats['loss_margins'] else 0
            
            # Performance consistency
            stats['score_std_dev'] = statistics.stdev(stats['captures_per_game']) if len(stats['captures_per_game']) > 1 else 0
            stats['min_score'] = min(stats['captures_per_game']) if stats['captures_per_game'] else 0
            stats['max_score'] = max(stats['captures_per_game']) if stats['captures_per_game'] else 0
            
            # Position-specific performance
            stats['games_as_agent0'] = len(stats['scores_when_agent0'])
            stats['games_as_agent1'] = len(stats['scores_when_agent1'])
            stats['avg_score_as_agent0'] = statistics.mean(stats['scores_when_agent0']) if stats['scores_when_agent0'] else 0
            stats['avg_score_as_agent1'] = statistics.mean(stats['scores_when_agent1']) if stats['scores_when_agent1'] else 0
            
            # Learning agent specific stats
            if stats['epsilon_values']:
                stats['initial_epsilon'] = max(stats['epsilon_values'])
                stats['final_epsilon'] = min(stats['epsilon_values'])
                stats['epsilon_decay_rate'] = (stats['initial_epsilon'] - stats['final_epsilon']) / len(stats['epsilon_values']) if len(stats['epsilon_values']) > 1 else 0
            else:
                stats['initial_epsilon'] = 'N/A'
                stats['final_epsilon'] = 'N/A'
                stats['epsilon_decay_rate'] = 'N/A'
        
        analysis_results.append(stats)
    
    return analysis_results


def write_analysis_to_csv(analysis_results, output_path):
    """Write analysis results to CSV file"""
    if not analysis_results:
        print("No analysis results to write.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Define the columns for the analysis CSV
    columns = [
        'agent_type', 'total_games', 'wins', 'losses', 'draws',
        'win_rate', 'loss_rate', 'draw_rate',
        'avg_score_per_game', 'avg_opponent_score', 'avg_score_differential',
        'avg_game_length', 'avg_total_moves',
        'avg_win_margin', 'avg_loss_margin', 'max_win_margin', 'max_loss_margin',
        'score_std_dev', 'min_score', 'max_score',
        'games_as_agent0', 'games_as_agent1', 'avg_score_as_agent0', 'avg_score_as_agent1',
        'initial_epsilon', 'final_epsilon', 'epsilon_decay_rate'
    ]
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        
        # Sort results by win rate (descending)
        sorted_results = sorted(analysis_results, key=lambda x: x['win_rate'], reverse=True)
        
        for stats in sorted_results:
            row = []
            for col in columns:
                value = stats.get(col, 'N/A')
                # Round floating point numbers to 4 decimal places
                if isinstance(value, float):
                    value = round(value, 4)
                row.append(value)
            writer.writerow(row)
    
    print(f"\nAnalysis complete! Results saved to {output_path}")


def analyze(sim_log_filename):
    """Main analysis function called from menu"""
    # Get the full path to the sim_log file
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sim_log_path = os.path.join(project_root, 'output', sim_log_filename)
    
    print("Performing deep statistical analysis...")
    analysis_results = analyze_sim_data(sim_log_path)
    
    if analysis_results:
        # Output to analysis_log.csv in the output directory
        output_path = os.path.join(project_root, 'output', 'analysis_log.csv')
        write_analysis_to_csv(analysis_results, output_path)
        
        # Print summary to console
        print(f"\n{'='*60}")
        print("AGENT PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        for stats in sorted(analysis_results, key=lambda x: x['win_rate'], reverse=True):
            print(f"\n{stats['agent_type']}:")
            print(f"  Games: {stats['total_games']} | Win Rate: {stats['win_rate']:.1%} | Avg Score: {stats['avg_score_per_game']:.1f}")
            print(f"  Score Differential: {stats['avg_score_differential']:+.1f} | Consistency (σ): {stats['score_std_dev']:.1f}")
            if stats['initial_epsilon'] != 'N/A':
                print(f"  Learning: ε {stats['initial_epsilon']:.3f} → {stats['final_epsilon']:.3f}")
        
        return analysis_results
    else:
        return None