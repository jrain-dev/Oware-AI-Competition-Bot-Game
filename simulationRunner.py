from owareEngine import OwareBoard
from agents import QLearningAgent, RandomAgent
from dataLogger import DataLogger
import random

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

        # Reset QLearningAgent episode state if needed
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

        # End episode for both agents
        if hasattr(agent0, "end_episode"):
            agent0.end_episode()
        if hasattr(agent1, "end_episode"):
            agent1.end_episode()

        # Early exit if one agent already won majority
        if max(wins) > best_of // 2:
            break
    return 0 if wins[0] > wins[1] else 1

def run_tournament():
    # Create 5 QLearningAgents and 5 RandomAgents
    contestants = [QLearningAgent() for _ in range(5)] + [RandomAgent() for _ in range(5)]
    random.shuffle(contestants)

    logger = DataLogger(filename="tournament_log.csv")

    # Quarterfinals: 5 matches, 10 agents
    print("Quarterfinals:")
    qf_winners = []
    for i in range(0, 10, 2):
        winner_idx = play_match(contestants[i], contestants[i+1], best_of=3)
        winner = contestants[i] if winner_idx == 0 else contestants[i+1]
        qf_winners.append(winner)
        logger.log_episode({
            "episode": f"QF{i//2+1}",
            "winner": type(winner).__name__,
            "score_p0": "",
            "score_p1": "",
            "epsilon": getattr(winner, "epsilon", "")
        })
        print(f"QF{i//2+1}: {type(contestants[i]).__name__} vs {type(contestants[i+1]).__name__} -> {type(winner).__name__}")

    # Semifinals: 2 matches, 4 agents (pick first 4 winners)
    print("\nSemifinals:")
    sf_winners = []
    for i in range(0, 4, 2):
        winner_idx = play_match(qf_winners[i], qf_winners[i+1], best_of=3)
        winner = qf_winners[i] if winner_idx == 0 else qf_winners[i+1]
        sf_winners.append(winner)
        logger.log_episode({
            "episode": f"SF{i//2+1}",
            "winner": type(winner).__name__,
            "score_p0": "",
            "score_p1": "",
            "epsilon": getattr(winner, "epsilon", "")
        })
        print(f"SF{i//2+1}: {type(qf_winners[i]).__name__} vs {type(qf_winners[i+1]).__name__} -> {type(winner).__name__}")

    # Finals: best of 5
    print("\nFinals:")
    winner_idx = play_match(sf_winners[0], sf_winners[1], best_of=5)
    winner = sf_winners[0] if winner_idx == 0 else sf_winners[1]
    logger.log_episode({
        "episode": "Final",
        "winner": type(winner).__name__,
        "score_p0": "",
        "score_p1": "",
        "epsilon": getattr(winner, "epsilon", "")
    })
    print(f"Final: {type(sf_winners[0]).__name__} vs {type(sf_winners[1]).__name__} -> {type(winner).__name__}")

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