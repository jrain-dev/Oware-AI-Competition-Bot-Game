"""
Comprehensive training module for deep learning models in Oware AI competition.

This module provides advanced training capabilities including:
- Configurable training parameters
- Progress tracking and evaluation
- Checkpoint management with versioning
- Training metrics and detailed logging
- Multi-opponent curriculum learning
"""

import os
import sys
import time
import json
import pickle
import random
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from owareEngine import OwareBoard
from agents import (Agent, RandomAgent, GreedyAgent, HeuristicAgent, 
                   MinimaxAgent, QLearningAgent, DQNSmall, DQNMedium, DQNLarge)

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Training episodes
        self.total_episodes = 5000
        self.warmup_episodes = 100  # Episodes before training starts
        
        # Evaluation settings
        self.eval_interval = 250  # Episodes between evaluations
        self.eval_episodes = 50   # Episodes per evaluation
        
        # Checkpoint settings  
        self.checkpoint_interval = 500  # Episodes between checkpoints
        self.save_best_only = True      # Only save if performance improves
        self.max_checkpoints = 5        # Maximum checkpoints to keep
        
        # Opponent curriculum
        self.opponent_types = ['random', 'greedy', 'heuristic', 'minimax']
        self.opponent_weights = [0.4, 0.3, 0.2, 0.1]  # Probability weights
        self.adaptive_curriculum = True  # Adapt opponent difficulty
        
        # Game settings
        self.variant = 'standard'
        
        # Logging
        self.log_interval = 100  # Episodes between progress logs
        self.detailed_logging = True
        
        # Early stopping
        self.patience = 1000  # Episodes without improvement before stopping
        self.min_improvement = 0.01  # Minimum win rate improvement
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


class TrainingMetrics:
    """Tracks and manages training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.losses = []
        self.exploration_rates = []
        self.evaluation_scores = {}
        self.best_win_rate = 0.0
        self.episodes_without_improvement = 0
        
    def add_episode(self, reward: float, length: int, win_rate: float = None):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if win_rate is not None:
            self.win_rates.append(win_rate)
    
    def add_loss(self, loss: float):
        """Add training loss."""
        self.losses.append(loss)
    
    def add_exploration_rate(self, epsilon: float):
        """Add exploration rate."""
        self.exploration_rates.append(epsilon)
    
    def add_evaluation(self, episode: int, results: Dict[str, float]):
        """Add evaluation results."""
        self.evaluation_scores[episode] = results
        
        # Check for improvement
        current_win_rate = results.get('overall_win_rate', 0.0)
        if current_win_rate > self.best_win_rate + 0.001:  # Small threshold for numerical stability
            self.best_win_rate = current_win_rate
            self.episodes_without_improvement = 0
            return True  # Improved
        else:
            self.episodes_without_improvement += 1
            return False  # No improvement
    
    def get_recent_average(self, metric: str, window: int = 100) -> float:
        """Get recent average of a metric."""
        data = getattr(self, metric, [])
        if not data:
            return 0.0
        return float(np.mean(data[-window:]))
    
    def save_to_file(self, filepath: str):
        """Save metrics to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load_from_file(self, filepath: str):
        """Load metrics from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)


class CheckpointManager:
    """Manages model checkpoints with versioning and best model tracking."""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.best_checkpoint = None
        self.best_score = -float('inf')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_checkpoint_history()
    
    def _load_checkpoint_history(self):
        """Load existing checkpoint history."""
        history_file = os.path.join(self.checkpoint_dir, 'checkpoint_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.checkpoints = data.get('checkpoints', [])
                self.best_checkpoint = data.get('best_checkpoint', None)
                self.best_score = data.get('best_score', -float('inf'))
    
    def _save_checkpoint_history(self):
        """Save checkpoint history."""
        history_file = os.path.join(self.checkpoint_dir, 'checkpoint_history.json')
        data = {
            'checkpoints': self.checkpoints,
            'best_checkpoint': self.best_checkpoint,
            'best_score': self.best_score
        }
        with open(history_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_checkpoint(self, agent: Agent, episode: int, score: float, 
                       metrics: TrainingMetrics, is_best: bool = False) -> str:
        """Save a checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_ep{episode}_{timestamp}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        # Save agent checkpoint if supported
        if hasattr(agent, 'save_checkpoint'):
            agent.save_checkpoint(checkpoint_path)
        
        # Save training metadata
        metadata_path = checkpoint_path.replace('.pth', '_metadata.json')
        metadata = {
            'episode': episode,
            'score': score,
            'timestamp': timestamp,
            'agent_type': type(agent).__name__,
            'best_win_rate': metrics.best_win_rate,
            'episodes_without_improvement': metrics.episodes_without_improvement
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save metrics
        metrics_path = checkpoint_path.replace('.pth', '_metrics.pkl')
        metrics.save_to_file(metrics_path)
        
        # Update checkpoint tracking
        checkpoint_info = {
            'episode': episode,
            'score': score,
            'path': checkpoint_path,
            'metadata_path': metadata_path,
            'metrics_path': metrics_path,
            'timestamp': timestamp
        }
        
        self.checkpoints.append(checkpoint_info)
        
        # Update best checkpoint
        if is_best or score > self.best_score:
            self.best_checkpoint = checkpoint_info
            self.best_score = score
            
            # Create symlink to best checkpoint
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            if os.path.exists(best_path):
                os.remove(best_path)
            os.symlink(os.path.basename(checkpoint_path), best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        self._save_checkpoint_history()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by episode (oldest first)
        self.checkpoints.sort(key=lambda x: x['episode'])
        
        # Remove oldest checkpoints
        to_remove = self.checkpoints[:-self.max_checkpoints]
        for checkpoint in to_remove:
            # Don't remove best checkpoint
            if checkpoint == self.best_checkpoint:
                continue
                
            # Remove files
            for path_key in ['path', 'metadata_path', 'metrics_path']:
                if path_key in checkpoint and os.path.exists(checkpoint[path_key]):
                    try:
                        os.remove(checkpoint[path_key])
                    except OSError:
                        pass
        
        # Update checkpoint list
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def load_best_checkpoint(self, agent: Agent) -> Optional[Dict]:
        """Load the best checkpoint."""
        if not self.best_checkpoint or not hasattr(agent, 'load_checkpoint'):
            return None
        
        checkpoint_path = self.best_checkpoint['path']
        if os.path.exists(checkpoint_path):
            agent.load_checkpoint(checkpoint_path)
            
            # Load metadata
            metadata_path = self.best_checkpoint['metadata_path']
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def get_checkpoint_summary(self) -> Dict:
        """Get summary of all checkpoints."""
        return {
            'total_checkpoints': len(self.checkpoints),
            'best_score': self.best_score,
            'best_checkpoint': self.best_checkpoint,
            'checkpoints': self.checkpoints
        }


class DQNTrainer:
    """Advanced trainer for DQN agents with comprehensive features."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.metrics = TrainingMetrics()
        
        # Create training directory
        self.training_dir = os.path.join('output', 'training')
        os.makedirs(self.training_dir, exist_ok=True)
        
        # Initialize components
        self.checkpoint_manager = None
        self.opponent_agents = self._create_opponent_agents()
        self.training_log = []
        
        # Training state
        self.current_episode = 0
        self.training_start_time = None
        
    def _create_opponent_agents(self) -> Dict[str, Agent]:
        """Create opponent agents for training."""
        opponents = {
            'random': RandomAgent(),
            'greedy': GreedyAgent(), 
            'heuristic': HeuristicAgent(),
            'minimax': MinimaxAgent(depth=2)
        }
        return opponents
    
    def _select_opponent(self) -> Agent:
        """Select an opponent based on curriculum."""
        opponent_type = np.random.choice(
            self.config.opponent_types,
            p=self.config.opponent_weights
        )
        return self.opponent_agents[opponent_type]
    
    def _play_training_game(self, agent: Agent, opponent: Agent, 
                           variant: str = 'standard') -> Tuple[int, int, Dict]:
        """Play a single training game."""
        board = OwareBoard(variant=variant)
        
        # Randomly assign player positions
        if random.random() < 0.5:
            agents = {0: agent, 1: opponent}
            agent_player = 0
        else:
            agents = {0: opponent, 1: agent}
            agent_player = 1
        
        state = board.reset()
        move_count = 0
        game_info = {'moves': [], 'rewards': []}
        
        while not board.game_over:
            current_player = board.current_player
            current_agent = agents[current_player]
            valid_moves = board.get_valid_moves(current_player)
            
            if not valid_moves:
                break
                
            action = current_agent.select_action(board, valid_moves)
            if action is None:
                break
            
            move_count += 1
            reward, next_state, done = board.apply_move(action)
            
            # Store experience and train if this is the learning agent
            if current_player == agent_player:
                if hasattr(agent, 'store'):
                    agent.store(state, action, reward, next_state, done)
                if hasattr(agent, 'train_step'):
                    agent.train_step()
                    
                game_info['moves'].append(action)
                game_info['rewards'].append(reward)
            
            state = next_state
        
        # Determine result from agent's perspective
        if board.winner == agent_player:
            result = 1  # Win
        elif board.winner == -1:
            result = 0  # Draw
        else:
            result = -1  # Loss
        
        # Call end_episode for learning agents
        if hasattr(agent, 'end_episode'):
            agent.end_episode()
        if hasattr(opponent, 'end_episode'):
            opponent.end_episode()
        
        game_info.update({
            'result': result,
            'final_scores': board.scores,
            'move_count': move_count,
            'agent_player': agent_player
        })
        
        return result, move_count, game_info
    
    def _evaluate_agent(self, agent: Agent, episode: int) -> Dict[str, float]:
        """Evaluate agent against all opponent types."""
        results = {}
        overall_wins = 0
        overall_games = 0
        
        for opp_type, opponent in self.opponent_agents.items():
            wins = 0
            games = self.config.eval_episodes
            
            for _ in range(games):
                result, _, _ = self._play_training_game(agent, opponent, self.config.variant)
                if result == 1:
                    wins += 1
                    overall_wins += 1
                overall_games += 1
            
            win_rate = wins / games
            results[f'{opp_type}_win_rate'] = win_rate
        
        results['overall_win_rate'] = overall_wins / overall_games if overall_games > 0 else 0.0
        return results
    
    def _log_progress(self, episode: int, recent_win_rate: float, 
                     epsilon: float = None, avg_loss: float = None):
        """Log training progress."""
        elapsed_time = time.time() - self.training_start_time
        episodes_per_second = episode / elapsed_time if elapsed_time > 0 else 0
        
        log_entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'episodes_per_second': episodes_per_second,
            'recent_win_rate': recent_win_rate,
            'epsilon': epsilon,
            'avg_loss': avg_loss,
            'best_win_rate': self.metrics.best_win_rate
        }
        
        self.training_log.append(log_entry)
        
        # Console output
        print(f"Episode {episode:6d} | "
              f"Win Rate: {recent_win_rate:.3f} | "
              f"Best: {self.metrics.best_win_rate:.3f} | "
              f"ε: {epsilon:.3f} | " if epsilon else "" +
              f"EPS: {episodes_per_second:.1f}")
    
    def _save_training_log(self):
        """Save training log to file."""
        log_path = os.path.join(self.training_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
    
    def train(self, agent_class, agent_name: str = None) -> Dict:
        """
        Train a DQN agent with comprehensive monitoring.
        
        Args:
            agent_class: Class of the agent to train (DQNSmall, DQNMedium, DQNLarge)
            agent_name: Optional name for the training session
            
        Returns:
            Dictionary with training results and statistics
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Cannot train DQN models.")
            return {}
        
        # Initialize training session
        if agent_name is None:
            agent_name = f"{agent_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session_dir = os.path.join(self.training_dir, agent_name)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save training config
        config_path = os.path.join(session_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Initialize checkpoint manager
        checkpoint_dir = os.path.join(session_dir, 'checkpoints')
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, self.config.max_checkpoints)
        
        # Create agent
        agent = agent_class()
        print(f"\nStarting training: {agent_name}")
        print(f"Agent: {agent_class.__name__}")
        print(f"Total episodes: {self.config.total_episodes}")
        print(f"Variant: {self.config.variant}")
        print("-" * 50)
        
        self.training_start_time = time.time()
        recent_results = deque(maxlen=100)
        
        try:
            for episode in range(1, self.config.total_episodes + 1):
                self.current_episode = episode
                
                # Select opponent
                opponent = self._select_opponent()
                
                # Play training game
                result, move_count, game_info = self._play_training_game(
                    agent, opponent, self.config.variant
                )
                
                recent_results.append(result)
                recent_win_rate = sum(1 for r in recent_results if r == 1) / len(recent_results)
                
                # Track metrics
                self.metrics.add_episode(result, move_count, recent_win_rate)
                if hasattr(agent, 'epsilon'):
                    self.metrics.add_exploration_rate(agent.epsilon)
                
                # Logging
                if episode % self.config.log_interval == 0:
                    epsilon = getattr(agent, 'epsilon', None)
                    self._log_progress(episode, recent_win_rate, epsilon)
                
                # Evaluation
                if episode % self.config.eval_interval == 0:
                    print(f"\nEvaluating at episode {episode}...")
                    eval_results = self._evaluate_agent(agent, episode)
                    improved = self.metrics.add_evaluation(episode, eval_results)
                    
                    print(f"Evaluation results:")
                    for opponent_type, win_rate in eval_results.items():
                        print(f"  {opponent_type}: {win_rate:.3f}")
                    
                    # Checkpointing
                    if episode % self.config.checkpoint_interval == 0:
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(
                            agent, episode, eval_results['overall_win_rate'], 
                            self.metrics, is_best=improved
                        )
                        print(f"Checkpoint saved: {os.path.basename(checkpoint_path)}")
                        if improved:
                            print("*** New best model! ***")
                
                # Early stopping check
                if (self.config.patience > 0 and 
                    self.metrics.episodes_without_improvement >= self.config.patience):
                    print(f"\nEarly stopping: No improvement for {self.config.patience} episodes")
                    break
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nTraining error: {e}")
            raise
        
        finally:
            # Final evaluation and cleanup
            print("\nFinal evaluation...")
            final_results = self._evaluate_agent(agent, self.current_episode)
            
            # Save final checkpoint
            if self.checkpoint_manager:
                self.checkpoint_manager.save_checkpoint(
                    agent, self.current_episode, final_results['overall_win_rate'],
                    self.metrics, is_best=False
                )
            
            # Save training log and metrics
            self._save_training_log()
            metrics_path = os.path.join(session_dir, 'final_metrics.pkl')
            self.metrics.save_to_file(metrics_path)
            
            # Training summary
            total_time = time.time() - self.training_start_time
            print(f"\nTraining completed!")
            print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
            print(f"Episodes completed: {self.current_episode}")
            print(f"Best win rate: {self.metrics.best_win_rate:.3f}")
            print(f"Final win rate: {final_results['overall_win_rate']:.3f}")
            print(f"Training data saved to: {session_dir}")
        
        return {
            'session_dir': session_dir,
            'episodes_completed': self.current_episode,
            'best_win_rate': self.metrics.best_win_rate,
            'final_win_rate': final_results['overall_win_rate'],
            'total_time': time.time() - self.training_start_time,
            'final_results': final_results,
            'checkpoint_summary': self.checkpoint_manager.get_checkpoint_summary() if self.checkpoint_manager else {}
        }


def create_training_config(**kwargs) -> TrainingConfig:
    """Create a training configuration with custom parameters."""
    config = TrainingConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}'")
    return config


def quick_train(agent_class, episodes: int = 2000, variant: str = 'standard', 
               name: str = None) -> Dict:
    """Quick training function with default settings."""
    config = create_training_config(
        total_episodes=episodes,
        variant=variant,
        eval_interval=max(50, episodes // 10),
        checkpoint_interval=max(100, episodes // 5)
    )
    
    trainer = DQNTrainer(config)
    return trainer.train(agent_class, name)


def load_trained_model(checkpoint_path: str, agent_class) -> Agent:
    """Load a trained model from checkpoint."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot load DQN models.")
        return None
    
    agent = agent_class()
    if hasattr(agent, 'load_checkpoint'):
        agent.load_checkpoint(checkpoint_path)
        print(f"Model loaded from: {checkpoint_path}")
        return agent
    else:
        print("Agent does not support checkpoint loading.")
        return None


def evaluate_trained_model(checkpoint_path: str, agent_class, 
                         episodes: int = 100, variant: str = 'standard') -> Dict:
    """Evaluate a trained model against all opponent types."""
    agent = load_trained_model(checkpoint_path, agent_class)
    if not agent:
        return {}
    
    # Create opponents
    opponents = {
        'random': RandomAgent(),
        'greedy': GreedyAgent(),
        'heuristic': HeuristicAgent(),
        'minimax': MinimaxAgent(depth=2)
    }
    
    results = {}
    print(f"\nEvaluating trained model against opponents ({episodes} games each):")
    
    for opp_name, opponent in opponents.items():
        wins = 0
        total_score = 0
        
        for _ in range(episodes):
            board = OwareBoard(variant=variant)
            
            # Randomly assign positions
            if random.random() < 0.5:
                agents = {0: agent, 1: opponent}
                agent_player = 0
            else:
                agents = {0: opponent, 1: agent}
                agent_player = 1
            
            # Play game
            state = board.reset()
            while not board.game_over:
                current_player = board.current_player
                current_agent = agents[current_player]
                valid_moves = board.get_valid_moves(current_player)
                
                if not valid_moves:
                    break
                    
                action = current_agent.select_action(board, valid_moves)
                if action is None:
                    break
                
                board.apply_move(action)
            
            # Check result
            if board.winner == agent_player:
                wins += 1
            
            total_score += board.scores[agent_player]
        
        win_rate = wins / episodes
        avg_score = total_score / episodes
        results[opp_name] = {
            'win_rate': win_rate,
            'avg_score': avg_score,
            'wins': wins,
            'games': episodes
        }
        
        print(f"  vs {opp_name:10s}: {win_rate:.3f} win rate, {avg_score:.1f} avg score")
    
    overall_wins = sum(r['wins'] for r in results.values())
    overall_games = sum(r['games'] for r in results.values())
    overall_win_rate = overall_wins / overall_games if overall_games > 0 else 0
    
    results['overall'] = {
        'win_rate': overall_win_rate,
        'wins': overall_wins,
        'games': overall_games
    }
    
    print(f"\nOverall performance: {overall_win_rate:.3f} win rate ({overall_wins}/{overall_games})")
    return results


def list_training_sessions(training_dir: str = None) -> List[Dict]:
    """List all available training sessions."""
    if training_dir is None:
        training_dir = os.path.join('output', 'training')
    
    if not os.path.exists(training_dir):
        return []
    
    sessions = []
    for session_name in os.listdir(training_dir):
        session_path = os.path.join(training_dir, session_name)
        if not os.path.isdir(session_path):
            continue
        
        # Check for config file
        config_path = os.path.join(session_path, 'config.json')
        checkpoint_dir = os.path.join(session_path, 'checkpoints')
        
        session_info = {
            'name': session_name,
            'path': session_path,
            'config_available': os.path.exists(config_path),
            'checkpoints_available': os.path.exists(checkpoint_dir)
        }
        
        # Try to get basic info from config
        if session_info['config_available']:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    session_info['total_episodes'] = config.get('total_episodes', 'Unknown')
                    session_info['variant'] = config.get('variant', 'Unknown')
            except:
                pass
        
        # Check for checkpoint history
        if session_info['checkpoints_available']:
            history_path = os.path.join(checkpoint_dir, 'checkpoint_history.json')
            if os.path.exists(history_path):
                try:
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                        session_info['best_score'] = history.get('best_score', 0)
                        session_info['checkpoint_count'] = len(history.get('checkpoints', []))
                except:
                    pass
        
        sessions.append(session_info)
    
    return sorted(sessions, key=lambda x: x['name'], reverse=True)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['small', 'medium', 'large']:
            # Training mode
            model_name = command
            if model_name == 'small':
                agent_class = DQNSmall
            elif model_name == 'medium':
                agent_class = DQNMedium
            elif model_name == 'large':
                agent_class = DQNLarge
            
            episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
            results = quick_train(agent_class, episodes=episodes)
            print(f"\nTraining completed with results: {results}")
            
        elif command == 'evaluate':
            # Evaluation mode
            if len(sys.argv) < 4:
                print("Usage: python training.py evaluate [checkpoint_path] [small|medium|large]")
                sys.exit(1)
            
            checkpoint_path = sys.argv[2]
            model_name = sys.argv[3].lower()
            
            if model_name == 'small':
                agent_class = DQNSmall
            elif model_name == 'medium':
                agent_class = DQNMedium
            elif model_name == 'large':
                agent_class = DQNLarge
            else:
                print("Invalid model type. Use: small, medium, or large")
                sys.exit(1)
            
            episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 100
            results = evaluate_trained_model(checkpoint_path, agent_class, episodes)
            
        elif command == 'list':
            # List training sessions
            sessions = list_training_sessions()
            if not sessions:
                print("No training sessions found.")
            else:
                print("Available training sessions:")
                for session in sessions:
                    print(f"  {session['name']}")
                    if 'total_episodes' in session:
                        print(f"    Episodes: {session['total_episodes']}, Variant: {session['variant']}")
                    if 'best_score' in session:
                        print(f"    Best score: {session['best_score']:.3f}, Checkpoints: {session.get('checkpoint_count', 0)}")
        else:
            print("Usage:")
            print("  python training.py [small|medium|large] [episodes]  - Train a model")
            print("  python training.py evaluate [checkpoint] [model] [episodes]  - Evaluate a model")
            print("  python training.py list  - List training sessions")
    else:
        print("Usage:")
        print("  python training.py [small|medium|large] [episodes]  - Train a model")
        print("  python training.py evaluate [checkpoint] [model] [episodes]  - Evaluate a model")
        print("  python training.py list  - List training sessions")