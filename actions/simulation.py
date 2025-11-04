import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from owareEngine import OwareBoard
from agents import QLearningAgent, RandomAgent, GreedyAgent, HeuristicAgent, MinimaxAgent, DQNSmall, DQNMedium, DQNLarge
from analysis import DataLogger
import random

def play_match(agent0, agent1, best_of=3):
    wins = [0, 0]
    match_stats = {
        "total_games": 0,
        "total_moves": 0,
        "total_scores": [0, 0],
        "game_lengths": [],
        "final_scores": []
    }
    
    for game_num in range(best_of):
        board = OwareBoard()
        agents = {0: agent0, 1: agent1}
        state = board.reset()
        done = False
        move_count = 0

        if hasattr(agent0, "end_episode"):
            agent0.end_episode()
        if hasattr(agent1, "end_episode"):
            agent1.end_episode()

        while not done:
            current_player_id = board.current_player
            current_agent = agents[current_player_id]
            valid_moves = board.get_valid_moves(current_player_id)
            action = current_agent.select_action(board, valid_moves)
            if action is None:
                break
            move_count += 1
            reward, next_state, done = board.apply_move(action)
            if current_player_id == 0 and hasattr(agent0, "update"):
                next_valid_moves = board.get_valid_moves(board.current_player)
                agent0.update(reward, next_state, next_valid_moves)
            state = next_state

        # Record game statistics
        match_stats["total_games"] += 1
        match_stats["total_moves"] += move_count
        match_stats["total_scores"][0] += board.scores[0]
        match_stats["total_scores"][1] += board.scores[1]
        match_stats["game_lengths"].append(move_count)
        match_stats["final_scores"].append((board.scores[0], board.scores[1]))

        if board.winner == 0:
            wins[0] += 1
        elif board.winner == 1:
            wins[1] += 1

        if hasattr(agent0, "end_episode"):
            agent0.end_episode()
        if hasattr(agent1, "end_episode"):
            agent1.end_episode()

        if max(wins) > best_of // 2:
            break
    
    winner_idx = 0 if wins[0] > wins[1] else 1
    return winner_idx, match_stats

def run_tournament():
    # Build contestants from all available agent classes
    raw_contestants = [
        RandomAgent(),
        GreedyAgent(),
        HeuristicAgent(),
        MinimaxAgent(),
        QLearningAgent(),
        DQNSmall(),
        DQNMedium(),
        DQNLarge(),
    ]

    # If odd number, add a RandomAgent to make it even
    if len(raw_contestants) % 2 == 1:
        raw_contestants.append(RandomAgent())

    contestants = []
    for idx, agent in enumerate(raw_contestants, start=1):
        contestants.append({
            "id": f"C{idx}",
            "agent": agent,
            "type": type(agent).__name__,
        })
    random.shuffle(contestants)

    logger = DataLogger(filename="tourney_log.csv")

    round_no = 1
    current = contestants
    # single-elimination rounds until one winner
    while len(current) > 1:
        next_round = []
        print(f"\nRound {round_no}: {len(current)} contestants")
        for i in range(0, len(current), 2):
            c0 = current[i]
            c1 = current[i+1]
            # final round uses best_of=5
            best_of = 5 if len(current) == 2 else 3
            winner_idx, match_stats = play_match(c0["agent"], c1["agent"], best_of=best_of)
            winner = c0 if winner_idx == 0 else c1
            next_round.append(winner)
            match_label = f"R{round_no}_M{i//2+1}"
            
            # Calculate match statistics
            avg_game_length = sum(match_stats["game_lengths"]) / len(match_stats["game_lengths"])
            total_captures_0 = match_stats["total_scores"][0]
            total_captures_1 = match_stats["total_scores"][1]
            score_diff = abs(total_captures_0 - total_captures_1)
            
            # Calculate winner margin across all games in the match
            if winner_idx == 0:
                winner_margin = total_captures_0 - total_captures_1
            else:
                winner_margin = total_captures_1 - total_captures_0
                
            logger.log_episode({
                "episode": match_label,
                "agent0_type": f"{c0['type']}({c0['id']})",
                "agent1_type": f"{c1['type']}({c1['id']})",
                "winner": f"{winner['type']}({winner['id']})",
                "agent0_score": total_captures_0,
                "agent1_score": total_captures_1,
                "total_moves": match_stats["total_moves"],
                "game_length": round(avg_game_length, 1),
                "agent0_captures": total_captures_0,
                "agent1_captures": total_captures_1,
                "final_board_seeds": "N/A",  # Could be enhanced to track across games
                "score_difference": score_diff,
                "winner_margin": winner_margin,
                "agent0_epsilon": getattr(c0["agent"], "epsilon", "N/A"),
                "agent1_epsilon": getattr(c1["agent"], "epsilon", "N/A")
            })
            print(f"{match_label}: {c0['type']}({c0['id']}) vs {c1['type']}({c1['id']}) -> {winner['type']}({winner['id']})")
        current = next_round
        round_no += 1

    champion = current[0]
    print(f"\nTournament finished. Champion: {champion['type']}({champion['id']}). Results saved to {logger.filename}")

def run_simulation(episodes):

    agent_factories = [
        RandomAgent,
        GreedyAgent,
        HeuristicAgent,
        lambda: MinimaxAgent(depth=2),
        QLearningAgent,
        DQNSmall,
        DQNMedium,
        DQNLarge,
    ]

    logger = DataLogger(filename="sim_log.csv")

    for episode in range(episodes):
        board = OwareBoard()
        # pick two random factories and instantiate agents
        f0 = random.choice(agent_factories)
        f1 = random.choice(agent_factories)
        a0 = f0() if callable(f0) else f0
        a1 = f1() if callable(f1) else f1
        agents = {0: a0, 1: a1}

        # call end_episode for agents that have it
        for a in agents.values():
            if hasattr(a, "end_episode"):
                a.end_episode()

        state = board.reset()
        done = False
        
        # Track game statistics
        move_count = 0
        initial_scores = [0, 0]
        
        while not done:
            current_player_id = board.current_player
            current_agent = agents[current_player_id]

            valid_moves = board.get_valid_moves(current_player_id)
            action = current_agent.select_action(board, valid_moves)

            if action is None:
                break

            # Track move count
            move_count += 1
            previous_scores = board.scores.copy()
            
            reward, next_state, done = board.apply_move(action)

            # apply learning updates for the agent that acted
            try:
                # Q-learning style
                if hasattr(current_agent, "update") and callable(current_agent.update):
                    next_valid_moves = board.get_valid_moves(board.current_player)
                    current_agent.update(reward, next_state, next_valid_moves)
                # DQN-like agents: store and train_step
                if hasattr(current_agent, "store"):
                    current_agent.store(state, action, reward, next_state, float(done))
                if hasattr(current_agent, "train_step"):
                    current_agent.train_step()
            except Exception:
                # don't let training errors stop the simulation
                pass

            state = next_state

        # determine winner label by agent types
        if board.winner == 0:
            winner_str = type(agents[0]).__name__
        elif board.winner == 1:
            winner_str = type(agents[1]).__name__
        else:
            winner_str = "Draw"

        # Calculate statistics
        final_board_seeds = sum(board.board)
        score_diff = abs(board.scores[0] - board.scores[1])
        agent0_captures = board.scores[0]
        agent1_captures = board.scores[1]
        
        # Determine winner margin (how decisively they won)
        if board.winner == 0:
            winner_margin = board.scores[0] - board.scores[1]
        elif board.winner == 1:
            winner_margin = board.scores[1] - board.scores[0]
        else:
            winner_margin = 0

        episode_summary = {
            "episode": episode + 1,
            "agent0_type": type(agents[0]).__name__,
            "agent1_type": type(agents[1]).__name__,
            "winner": winner_str,
            "agent0_score": board.scores[0],
            "agent1_score": board.scores[1],
            "total_moves": move_count,
            "game_length": move_count,  # Same as total_moves for now
            "agent0_captures": agent0_captures,
            "agent1_captures": agent1_captures,
            "final_board_seeds": final_board_seeds,
            "score_difference": score_diff,
            "winner_margin": winner_margin,
            "agent0_epsilon": getattr(agents[0], "epsilon", "N/A"),
            "agent1_epsilon": getattr(agents[1], "epsilon", "N/A")
        }

        logger.log_episode(episode_summary)

        # call end_episode hooks
        for a in agents.values():
            if hasattr(a, "end_episode"):
                try:
                    a.end_episode()
                except Exception:
                    pass

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} complete.")

    print(f"\nSimulation finished. Results saved to {logger.filename}")


if __name__ == "__main__":
    run_simulation()