"""
Statistics recorder for Oware AI agents.

This module provides utilities for collecting and recording metrics about
agent decision-making, including timing, node expansion, and evaluation scores.
"""

import json
import csv
import time
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from statistics import mean, median


@dataclass
class MoveMetrics:
    """Metrics for a single move decision."""
    move_number: int
    player: int
    decision_time: float
    nodes_expanded: Optional[int] = None
    evaluation_score: Optional[float] = None
    move_selected: Optional[int] = None
    valid_moves: Optional[List[int]] = None


@dataclass
class MatchMetrics:
    """Complete metrics for a single match."""
    match_id: str
    timestamp: str
    duration_seconds: float
    moves_count: int
    avg_decision_time: float
    median_decision_time: float
    decision_times: List[float]
    total_nodes_expanded: Optional[int]
    avg_nodes_expanded_per_move: Optional[float]
    evaluations: List[Optional[float]]
    final_board_state: List[int]
    final_scores: List[int]
    winner: Optional[int]
    agent0_info: Dict[str, Any]
    agent1_info: Dict[str, Any]
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StatsRecorder:
    """
    Thread-safe recorder for match statistics.
    
    Writes per-match data to JSONL and optionally CSV format.
    """
    
    def __init__(self, output_path: str = "stats.jsonl", 
                 enable_csv: bool = False,
                 warn_on_error: bool = True):
        """
        Initialize the stats recorder.
        
        Args:
            output_path: Path to JSONL output file (default: stats.jsonl)
            enable_csv: Also write to CSV format (default: False)
            warn_on_error: Print warnings on file errors (default: True)
        """
        self.output_path = Path(output_path)
        self.csv_path = self.output_path.with_suffix('.csv') if enable_csv else None
        self.warn_on_error = warn_on_error
        self.enabled = True
        self.lock = threading.Lock()
        
        # Test if we can write to the output path
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Test write
            with open(self.output_path, 'a') as f:
                pass
        except (IOError, OSError) as e:
            if self.warn_on_error:
                print(f"Warning: Cannot write to {self.output_path}: {e}")
                print("Stats recording will be disabled.")
            self.enabled = False
        
        # Initialize CSV if requested
        if self.csv_path and self.enabled:
            try:
                self._init_csv()
            except (IOError, OSError) as e:
                if self.warn_on_error:
                    print(f"Warning: Cannot write CSV to {self.csv_path}: {e}")
                self.csv_path = None
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_path:
            return
            
        headers = [
            'match_id', 'timestamp', 'duration_seconds', 'moves_count',
            'avg_decision_time', 'median_decision_time', 
            'total_nodes_expanded', 'avg_nodes_expanded_per_move',
            'winner', 'agent0_name', 'agent0_type', 
            'agent1_name', 'agent1_type', 'seed'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def record_match(self, metrics: MatchMetrics) -> bool:
        """
        Record a completed match to the output files.
        
        Args:
            metrics: MatchMetrics object containing all match data
            
        Returns:
            True if successfully recorded, False otherwise
        """
        if not self.enabled:
            return False
        
        with self.lock:
            try:
                # Write JSONL - convert to dict and handle numpy types
                data = asdict(metrics)
                # Convert any numpy arrays to lists
                data = self._ensure_json_serializable(data)
                
                with open(self.output_path, 'a') as f:
                    json.dump(data, f)
                    f.write('\n')
                
                # Write CSV if enabled
                if self.csv_path:
                    self._append_csv(metrics)
                
                return True
            except (IOError, OSError) as e:
                if self.warn_on_error:
                    print(f"Warning: Error writing stats: {e}")
                return False
    
    def _ensure_json_serializable(self, obj):
        """Recursively convert non-serializable objects to JSON-compatible types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _append_csv(self, metrics: MatchMetrics):
        """Append a row to the CSV file."""
        if not self.csv_path:
            return
        
        row = [
            metrics.match_id,
            metrics.timestamp,
            f"{metrics.duration_seconds:.4f}",
            metrics.moves_count,
            f"{metrics.avg_decision_time:.6f}",
            f"{metrics.median_decision_time:.6f}",
            metrics.total_nodes_expanded if metrics.total_nodes_expanded is not None else '',
            f"{metrics.avg_nodes_expanded_per_move:.2f}" if metrics.avg_nodes_expanded_per_move is not None else '',
            metrics.winner if metrics.winner is not None else 'Draw',
            metrics.agent0_info.get('name', ''),
            metrics.agent0_info.get('type', ''),
            metrics.agent1_info.get('name', ''),
            metrics.agent1_info.get('type', ''),
            metrics.seed if metrics.seed is not None else '',
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


class DecisionTimer:
    """
    Context manager for timing agent decisions and collecting metrics.
    
    Usage:
        with DecisionTimer() as timer:
            move = agent.select_action(state, valid_moves)
        
        metrics = timer.get_metrics(
            move_number=1, 
            player=0, 
            move_selected=move,
            valid_moves=valid_moves
        )
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.nodes_expanded: Optional[int] = None
        self.evaluation_score: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        return False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    def set_nodes_expanded(self, count: int):
        """Set the number of nodes expanded during decision."""
        self.nodes_expanded = count
    
    def set_evaluation_score(self, score: float):
        """Set the evaluation score returned by the agent."""
        self.evaluation_score = score
    
    def get_metrics(self, move_number: int, player: int, 
                   move_selected: Optional[int] = None,
                   valid_moves: Optional[List[int]] = None) -> MoveMetrics:
        """
        Create a MoveMetrics object from the collected data.
        
        Args:
            move_number: The move number in the game
            player: The player ID (0 or 1)
            move_selected: The move that was selected
            valid_moves: List of valid moves available
            
        Returns:
            MoveMetrics object
        """
        return MoveMetrics(
            move_number=move_number,
            player=player,
            decision_time=self.elapsed,
            nodes_expanded=self.nodes_expanded,
            evaluation_score=self.evaluation_score,
            move_selected=move_selected,
            valid_moves=valid_moves
        )


class MatchRecorder:
    """
    Helper class to track metrics for a single match.
    
    Usage:
        recorder = MatchRecorder(match_id="match_001", agent0_info=..., agent1_info=...)
        
        # During game loop
        with recorder.time_move(move_number, player) as timer:
            move = agent.select_action(state, valid_moves)
            timer.set_nodes_expanded(agent.nodes_expanded)
        
        # At match end
        metrics = recorder.finalize(board)
        stats_recorder.record_match(metrics)
    """
    
    def __init__(self, match_id: str, agent0_info: Dict[str, Any], 
                 agent1_info: Dict[str, Any], seed: Optional[int] = None):
        """
        Initialize a match recorder.
        
        Args:
            match_id: Unique identifier for the match
            agent0_info: Dict with agent 0 information (name, type, params, etc.)
            agent1_info: Dict with agent 1 information
            seed: Optional random seed used for the match
        """
        self.match_id = match_id
        self.agent0_info = agent0_info
        self.agent1_info = agent1_info
        self.seed = seed
        self.start_time = time.perf_counter()
        self.move_metrics: List[MoveMetrics] = []
        self.metadata: Dict[str, Any] = {}
    
    @contextmanager
    def time_move(self, move_number: int, player: int):
        """
        Context manager for timing a single move.
        
        Args:
            move_number: The move number
            player: The player making the move (0 or 1)
            
        Yields:
            DecisionTimer that can be used to record additional metrics
        """
        timer = DecisionTimer()
        with timer:
            yield timer
        
        # Note: caller should call record_move() after to save the metrics
    
    def record_move(self, metrics: MoveMetrics):
        """Record metrics for a completed move."""
        self.move_metrics.append(metrics)
    
    def finalize(self, board_state: List[int], scores: List[int], 
                winner: Optional[int]) -> MatchMetrics:
        """
        Finalize the match and create a MatchMetrics object.
        
        Args:
            board_state: Final board state as list of pit values
            scores: Final scores [player0_score, player1_score]
            winner: Winner ID (0, 1, or None for draw)
            
        Returns:
            MatchMetrics object ready to be recorded
        """
        duration = time.perf_counter() - self.start_time
        
        decision_times = [m.decision_time for m in self.move_metrics]
        nodes_list = [m.nodes_expanded for m in self.move_metrics if m.nodes_expanded is not None]
        evaluations = [m.evaluation_score for m in self.move_metrics]
        
        return MatchMetrics(
            match_id=self.match_id,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            moves_count=len(self.move_metrics),
            avg_decision_time=mean(decision_times) if decision_times else 0.0,
            median_decision_time=median(decision_times) if decision_times else 0.0,
            decision_times=decision_times,
            total_nodes_expanded=sum(nodes_list) if nodes_list else None,
            avg_nodes_expanded_per_move=mean(nodes_list) if nodes_list else None,
            evaluations=evaluations,
            final_board_state=board_state,
            final_scores=scores,
            winner=winner,
            agent0_info=self.agent0_info,
            agent1_info=self.agent1_info,
            seed=self.seed,
            metadata=self.metadata
        )
