import numpy as np
import random
from collections import defaultdict

class Agent:
    def select_action(self, state, valid_moves):
        raise NotImplementedError


class RandomAgent(Agent):
    def select_action(self, state, valid_moves):
        if not valid_moves:
            return None
        return random.choice(valid_moves)


class QLearningAgent(Agent):
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.999, min_exploration_rate=0.01):
        self.q_table = defaultdict(lambda: np.zeros(12))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_epsilon = min_exploration_rate
        self.last_state = None
        self.last_action = None

    def select_action(self, state, valid_moves):
        if not valid_moves:
            return None

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(valid_moves)
        else:
            q_values = self.q_table[state]
            valid_q_values = {move: q_values[move] for move in valid_moves}
            action = max(valid_q_values, key=valid_q_values.get)

        self.last_state = state
        self.last_action = action
        return action

    def update(self, reward, new_state, new_valid_moves):
        if self.last_state is None or self.last_action is None:
            return

        old_value = self.q_table[self.last_state][self.last_action]

        if not new_valid_moves:
            next_max = 0
        else:
            next_max = np.max(self.q_table[new_state])

        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[self.last_state][self.last_action] = new_value

    def end_episode(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.exploration_decay)
        self.last_state = None
        self.last_action = None
