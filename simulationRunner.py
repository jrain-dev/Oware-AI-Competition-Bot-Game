from owareEngine import OwareBoard
from agents import QLearningAgent, RandomAgent
from dataLogger import DataLogger
import random

def play_match(agent0, agent1, best_of=3):
    wins = [0, 0]
    for _ in range(best_of):
        board = OwareBoard()
        agents = {0: agent0, 1: agent1}
        state = board.reset()
        done = False

        if hasattr(agent0, "end_episode"):
            agent0.end_episode()
        if hasattr(agent1, "end_episode"):
            agent1.end_episode()

        while not done:
            current_player_id = board.current_player
            current_agent = agents[current_player_id]
            valid_moves = board.get_valid_moves(current_player_id)
            action = current_agent.select_action(state, valid_moves)
            if action is None:
                break
            reward, next_state, done = board.apply_move(action)
            if current_player_id == 0 and hasattr(agent0, "update"):
                next_valid_moves = board.get_valid_moves(board.current_player)
                agent0.update(reward, next_state, next_valid_moves)
            state = next_state

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
    return 0 if wins[0] > wins[1] else 1

def run_tournament():
    # 5 QLearningAgents and 5 RandomAgents
    raw_contestants = [QLearningAgent() for _ in range(5)] + [RandomAgent() for _ in range(5)]
    contestants = []
    for idx, agent in enumerate(raw_contestants, start=1):
        contestants.append({
            "id": f"C{idx}",
            "agent": agent,
            "type": type(agent).__name__,
        })
    random.shuffle(contestants)

    columns = [
        "Match",
        "Contestant0_ID",
        "Contestant0_Type",
        "Contestant1_ID",
        "Contestant1_Type",
        "Winner_ID",
        "Winner_Type",
    ]
    data_keys = [
        "episode",
        "c0_id",
        "c0_type",
        "c1_id",
        "c1_type",
        "winner_id",
        "winner_type",
    ]
    logger = DataLogger(filename="tournament_log.csv", columns=columns, data_keys=data_keys)

    # 5 matches, 10 agents
    print("Quarterfinals:")
    qf_winners = []
    for i in range(0, 10, 2):
        c0 = contestants[i]
        c1 = contestants[i+1]
        winner_idx = play_match(c0["agent"], c1["agent"], best_of=3)
        winner = c0 if winner_idx == 0 else c1
        qf_winners.append(winner)
        logger.log_episode({
            "episode": f"QF{i//2+1}",
            "c0_id": c0["id"],
            "c0_type": c0["type"],
            "c1_id": c1["id"],
            "c1_type": c1["type"],
            "winner_id": winner["id"],
            "winner_type": winner["type"],
        })
        print(f"QF{i//2+1}: {c0['type']}({c0['id']}) vs {c1['type']}({c1['id']}) -> {winner['type']}({winner['id']})")

    # 2 matches, 4 agents 
    print("\nSemifinals:")
    sf_winners = []
    for i in range(0, 4, 2):
        c0 = qf_winners[i]
        c1 = qf_winners[i+1]
        winner_idx = play_match(c0["agent"], c1["agent"], best_of=3)
        winner = c0 if winner_idx == 0 else c1
        sf_winners.append(winner)
        logger.log_episode({
            "episode": f"SF{i//2+1}",
            "c0_id": c0["id"],
            "c0_type": c0["type"],
            "c1_id": c1["id"],
            "c1_type": c1["type"],
            "winner_id": winner["id"],
            "winner_type": winner["type"],
        })
        print(f"SF{i//2+1}: {c0['type']}({c0['id']}) vs {c1['type']}({c1['id']}) -> {winner['type']}({winner['id']})")

    # best of 5
    print("\nFinals:")
    c0 = sf_winners[0]
    c1 = sf_winners[1]
    winner_idx = play_match(c0["agent"], c1["agent"], best_of=5)
    winner = c0 if winner_idx == 0 else c1
    logger.log_episode({
        "episode": "Final",
        "c0_id": c0["id"],
        "c0_type": c0["type"],
        "c1_id": c1["id"],
        "c1_type": c1["type"],
        "winner_id": winner["id"],
        "winner_type": winner["type"],
    })
    print(f"Final: {c0['type']}({c0['id']}) vs {c1['type']}({c1['id']}) -> {winner['type']}({winner['id']})")

    print(f"\nTournament finished. Results saved to {logger.filename}")

def run_simulation(episodes=10000):
    board = OwareBoard()
    player0 = QLearningAgent()
    player1 = RandomAgent()
    agents = {0: player0, 1: player1}

    logger = DataLogger(filename="training_log.csv")

    for episode in range(episodes):
        state = board.reset()
        done = False

        while not done:
            current_player_id = board.current_player
            current_agent = agents[current_player_id]

            valid_moves = board.get_valid_moves(current_player_id)
            action = current_agent.select_action(state, valid_moves)

            if action is None:
                break

            reward, next_state, done = board.apply_move(action)

            if current_player_id == 0:
                next_valid_moves = board.get_valid_moves(board.current_player)
                player0.update(reward, next_state, next_valid_moves)

        winner = "Draw"
        if board.winner == 0:
            winner = "Player0_QLearning"
        elif board.winner == 1:
            winner = "Player1_Random"

        episode_summary = {
            "episode": episode + 1,
            "winner": winner,
            "score_p0": board.scores[0],
            "score_p1": board.scores[1],
            "epsilon": player0.epsilon,
        }
        logger.log_episode(episode_summary)

        player0.end_episode()

        if (episode + 1) % 100 == 0:
            print(
                f"Episode {episode + 1}/{episodes} complete. P0 Epsilon: {player0.epsilon:.4f}"
            )

    print(f"\nSimulation finished. Results saved to {logger.filename}")


if __name__ == "__main__":
    run_tournament()
