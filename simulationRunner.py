from owareEngine import OwareBoard
from agents import QLearningAgent, RandomAgent
from dataLogger import DataLogger


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
    run_simulation(episodes=20000)