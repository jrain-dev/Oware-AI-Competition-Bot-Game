import numpy as np
import random
from collections import defaultdict
import copy
import os
import pickle


class Agent:
    def select_action(self, board, valid_moves):
        raise NotImplementedError


class RandomAgent(Agent):
    def select_action(self, board, valid_moves):
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

    def select_action(self, board, valid_moves):
        if not valid_moves:
            return None

        state = board.get_state()

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

    def save_checkpoint(self, path):
        """Save the Q-table and some hyperparameters to a pickle file."""
        data = {
            'q_table': dict(self.q_table),
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
        }
        # ensure directory exists
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_checkpoint(self, path):
        """Load the Q-table and hyperparameters from a pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(12))
        # update q_table entries
        qd = data.get('q_table', {})
        # dict keys should match whichever state representation was used.
        for k, v in qd.items():
            self.q_table[k] = np.array(v, dtype=np.float32)
        self.lr = data.get('lr', self.lr)
        self.gamma = data.get('gamma', self.gamma)
        self.epsilon = data.get('epsilon', self.epsilon)


class GreedyAgent(Agent):
    """Chooses the move that results in the highest immediate score gain by simulating the move."""

    def select_action(self, board, valid_moves):
        if not valid_moves:
            return None
        best_move = None
        best_gain = -float("inf")
        for m in valid_moves:
            b_copy = copy.deepcopy(board)
            # apply move and inspect score difference
            before = sum(b_copy.scores)
            b_copy.apply_move(m)
            after = sum(b_copy.scores)
            gain = after - before
            if gain > best_gain:
                best_gain = gain
                best_move = m
        return best_move


class HeuristicAgent(Agent):
    """Prefers capturing moves; otherwise minimizes opponent immediate capture opportunities."""

    def select_action(self, board, valid_moves):
        if not valid_moves:
            return None
        best_move = None
        best_score = -float("inf")
        for m in valid_moves:
            b_copy = copy.deepcopy(board)
            before = b_copy.scores[b_copy.current_player]
            b_copy.apply_move(m)
            after = b_copy.scores[b_copy.current_player]
            capture_gain = after - before
            # if capture occurs, prefer it strongly
            if capture_gain > 0:
                score = 100 + capture_gain
            else:
                # evaluate opponent's best immediate gain next turn; we want to minimize it
                opp_moves = b_copy.get_valid_moves(b_copy.current_player)
                opp_best = 0
                for om in opp_moves:
                    b2 = copy.deepcopy(b_copy)
                    b2.apply_move(om)
                    opp_gain = b2.scores[b2.current_player] - b_copy.scores[b_copy.current_player]
                    opp_best = max(opp_best, opp_gain)
                score = -opp_best
            if score > best_score:
                best_score = score
                best_move = m
        return best_move


class MinimaxAgent(Agent):
    """Simple depth-limited minimax agent with heuristic based on score difference."""

    def __init__(self, depth=2):
        self.depth = depth

    def select_action(self, board, valid_moves):
        if not valid_moves:
            return None

        best_move = None
        best_val = -float("inf")
        for m in valid_moves:
            b_copy = copy.deepcopy(board)
            b_copy.apply_move(m)
            val = self._minimax(b_copy, self.depth - 1, maximizing=False)
            if val > best_val:
                best_val = val
                best_move = m
        return best_move

    def _minimax(self, board, depth, maximizing):
        if depth == 0 or board.game_over:
            return board.scores[0] - board.scores[1]
        current = board.current_player
        moves = board.get_valid_moves(current)
        if not moves:
            return board.scores[0] - board.scores[1]
        if maximizing:
            val = -float("inf")
            for m in moves:
                b_copy = copy.deepcopy(board)
                b_copy.apply_move(m)
                val = max(val, self._minimax(b_copy, depth - 1, False))
            return val
        else:
            val = float("inf")
            for m in moves:
                b_copy = copy.deepcopy(board)
                b_copy.apply_move(m)
                val = min(val, self._minimax(b_copy, depth - 1, True))
            return val


# -----------------------------
# Optional DQN agents (require PyTorch)
# -----------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    from collections import deque
    import random as _random

    class DQNNet(nn.Module):
        def __init__(self, input_dim=12, hidden_sizes=(64, 64), output_dim=12):
            super().__init__()
            layers = []
            last = input_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(last, h))
                layers.append(nn.ReLU())
                last = h
            layers.append(nn.Linear(last, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class DQNAgent(Agent):
        """A simple DQN agent with experience replay and target network."""

        def __init__(self, hidden_sizes=(64, 64), lr=1e-3, gamma=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, batch_size=64, buffer_size=10000, target_update=10, device=None):
            self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self.net = DQNNet(input_dim=12, hidden_sizes=hidden_sizes).to(self.device)
            self.target = DQNNet(input_dim=12, hidden_sizes=hidden_sizes).to(self.device)
            self.target.load_state_dict(self.net.state_dict())
            self.opt = optim.Adam(self.net.parameters(), lr=lr)
            self.gamma = gamma
            self.epsilon = exploration_rate
            self.epsilon_decay = exploration_decay
            self.min_epsilon = min_exploration_rate
            self.batch_size = batch_size
            self.buffer = deque(maxlen=buffer_size)
            self.target_update = target_update
            self.step_count = 0

        def _state_tensor(self, state):
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            # normalize by median seeds (safe): divide by 6
            x = x / 6.0
            return x.unsqueeze(0)

        def select_action(self, board, valid_moves):
            if not valid_moves:
                return None
            # epsilon-greedy
            if _random.random() < self.epsilon:
                return _random.choice(valid_moves)
            s = self._state_tensor(board.get_state())
            with torch.no_grad():
                q = self.net(s).cpu().numpy().flatten()
            # mask invalid moves by setting them to -inf
            mask = [-1e9] * 12
            for m in valid_moves:
                mask[m] = q[m]
            # choose argmax among masked
            best = int(max(range(12), key=lambda i: mask[i]))
            return best

        def store(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))

        def train_step(self):
            if len(self.buffer) < self.batch_size:
                return
            batch = _random.sample(self.buffer, self.batch_size)
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device) / 6.0
            actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device) / 6.0
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device).unsqueeze(1)

            q_values = self.net(states).gather(1, actions)
            with torch.no_grad():
                next_q = self.target(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + (1.0 - dones) * (self.gamma * next_q)

            loss = nn.functional.mse_loss(q_values, target_q)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            self.step_count += 1
            if self.step_count % self.target_update == 0:
                self.target.load_state_dict(self.net.state_dict())

        def end_episode(self):
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)



        def save_checkpoint(self, path):
            """Save PyTorch model, optimizer state and exploration rate to path."""
            import torch
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            torch.save({
                'net_state': self.net.state_dict(),
                'target_state': self.target.state_dict(),
                'opt_state': self.opt.state_dict(),
                'epsilon': self.epsilon,
            }, path)

        def load_checkpoint(self, path):
            """Load PyTorch model/optimizer state from path."""
            import torch
            data = torch.load(path, map_location=self.device)
            self.net.load_state_dict(data.get('net_state', {}))
            target_state = data.get('target_state', None)
            if target_state is not None:
                self.target.load_state_dict(target_state)
            opt_state = data.get('opt_state', None)
            if opt_state is not None:
                try:
                    self.opt.load_state_dict(opt_state)
                except Exception:
                    # optimizer state may be incompatible across devices/versions; ignore safely
                    pass
            self.epsilon = data.get('epsilon', self.epsilon)


    # Convenience factories for three DQN variants
    class DQNSmall(DQNAgent):
        def __init__(self):
            super().__init__(hidden_sizes=(64,), lr=1e-3, batch_size=32, buffer_size=5000, target_update=20)


    class DQNMedium(DQNAgent):
        def __init__(self):
            super().__init__(hidden_sizes=(128, 64), lr=5e-4, batch_size=64, buffer_size=10000, target_update=10)


    class DQNLarge(DQNAgent):
        def __init__(self):
            super().__init__(hidden_sizes=(256, 128), lr=3e-4, batch_size=128, buffer_size=20000, target_update=5)

else:
    # If PyTorch isn't available, provide a numpy-based DQN implementation so DQN agents still work.
    class NumpyDQNAgent(Agent):
        def __init__(self, hidden_sizes=(64, 64), lr=1e-3, gamma=0.99, exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01, batch_size=64, buffer_size=10000, target_update=10):
            self.gamma = gamma
            self.epsilon = exploration_rate
            self.epsilon_decay = exploration_decay
            self.min_epsilon = min_exploration_rate
            self.batch_size = batch_size
            self.buffer = []
            self.buffer_size = buffer_size
            self.target_update = target_update
            self.step_count = 0

            # network dimensions
            self.input_dim = 12
            self.output_dim = 12
            hs = list(hidden_sizes)
            if len(hs) == 0:
                hs = [64]
            # initialize weights (W1,b1,W2,b2) for a single hidden layer or two layers
            # we'll support up to two hidden layers
            if len(hs) == 1:
                h1 = hs[0]
                self.W1 = np.random.randn(self.input_dim, h1).astype(np.float32) * 0.01
                self.b1 = np.zeros(h1, dtype=np.float32)
                self.W2 = np.random.randn(h1, self.output_dim).astype(np.float32) * 0.01
                self.b2 = np.zeros(self.output_dim, dtype=np.float32)
                # target weights
                self.tW1 = self.W1.copy()
                self.tb1 = self.b1.copy()
                self.tW2 = self.W2.copy()
                self.tb2 = self.b2.copy()
            else:
                # support two hidden layers
                h1, h2 = hs[0], hs[1]
                self.W1 = np.random.randn(self.input_dim, h1).astype(np.float32) * 0.01
                self.b1 = np.zeros(h1, dtype=np.float32)
                self.W2 = np.random.randn(h1, h2).astype(np.float32) * 0.01
                self.b2 = np.zeros(h2, dtype=np.float32)
                self.W3 = np.random.randn(h2, self.output_dim).astype(np.float32) * 0.01
                self.b3 = np.zeros(self.output_dim, dtype=np.float32)
                # target
                self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
                self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()
                self.tW3 = self.W3.copy(); self.tb3 = self.b3.copy()

            self.lr = lr

        def _state_array(self, state):
            # state can be tuple or np array
            s = np.array(state, dtype=np.float32).reshape(-1)
            return s / 6.0

        def _forward(self, states):
            # states: (batch, input_dim)
            if hasattr(self, 'W3'):
                z1 = states.dot(self.W1) + self.b1
                a1 = np.maximum(0, z1)
                z2 = a1.dot(self.W2) + self.b2
                a2 = np.maximum(0, z2)
                z3 = a2.dot(self.W3) + self.b3
                return z1, a1, z2, a2, z3
            else:
                z1 = states.dot(self.W1) + self.b1
                a1 = np.maximum(0, z1)
                z2 = a1.dot(self.W2) + self.b2
                return z1, a1, z2

        def _target_forward(self, states):
            if hasattr(self, 'tW3'):
                z1 = states.dot(self.tW1) + self.tb1
                a1 = np.maximum(0, z1)
                z2 = a1.dot(self.tW2) + self.tb2
                a2 = np.maximum(0, z2)
                z3 = a2.dot(self.tW3) + self.tb3
                return z3
            else:
                z1 = states.dot(self.tW1) + self.tb1
                a1 = np.maximum(0, z1)
                z2 = a1.dot(self.tW2) + self.tb2
                return z2

        def select_action(self, board, valid_moves):
            if not valid_moves:
                return None
            if random.random() < self.epsilon:
                return random.choice(valid_moves)
            s = self._state_array(board.get_state()).reshape(1, -1)
            if hasattr(self, 'W3'):
                _, _, _, _, q = self._forward(s)
            else:
                _, _, q = self._forward(s)
            q = q.flatten()
            # mask invalid
            mask = np.full_like(q, -1e9)
            for m in valid_moves:
                mask[m] = q[m]
            return int(np.argmax(mask))

        def store(self, state, action, reward, next_state, done):
            if len(self.buffer) >= self.buffer_size:
                self.buffer.pop(0)
            self.buffer.append((np.array(state, dtype=np.float32), int(action), float(reward), np.array(next_state, dtype=np.float32), float(done)))

        def train_step(self):
            if len(self.buffer) < 2:
                return
            batch_size = min(self.batch_size, len(self.buffer))
            batch = random.sample(self.buffer, batch_size)
            states = np.stack([self._state_array(b[0]) for b in batch])
            actions = np.array([b[1] for b in batch], dtype=np.int64)
            rewards = np.array([b[2] for b in batch], dtype=np.float32)
            next_states = np.stack([self._state_array(b[3]) for b in batch])
            dones = np.array([b[4] for b in batch], dtype=np.float32)

            # compute current Q
            if hasattr(self, 'W3'):
                z1, a1, z2, a2, q_preds = self._forward(states)
                next_q = self._target_forward(next_states)
            else:
                z1, a1, q_preds = self._forward(states)
                next_q = self._target_forward(next_states)

            q_preds = q_preds.astype(np.float32)
            max_next = np.max(next_q, axis=1)
            targets = rewards + (1.0 - dones) * (self.gamma * max_next)

            # compute gradient for output layer
            batch_n = states.shape[0]
            # create g2: (batch, output_dim)
            g2 = np.zeros_like(q_preds)
            g2[np.arange(batch_n), actions] = 2.0 * (q_preds[np.arange(batch_n), actions] - targets)
            g2 = g2 / batch_n

            if hasattr(self, 'W3'):
                # grads for W3,b3
                dW3 = a2.T.dot(g2)
                db3 = np.sum(g2, axis=0)
                # backprop to a2
                da2 = g2.dot(self.W3.T)
                dz2 = da2 * (z2 > 0)
                dW2 = a1.T.dot(dz2)
                db2 = np.sum(dz2, axis=0)
                da1 = dz2.dot(self.W2.T)
                dz1 = da1 * (z1 > 0)
                dW1 = states.T.dot(dz1)
                db1 = np.sum(dz1, axis=0)

                # update weights
                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
            else:
                dW2 = a1.T.dot(g2)
                db2 = np.sum(g2, axis=0)
                da1 = g2.dot(self.W2.T)
                dz1 = da1 * (z1 > 0)
                dW1 = states.T.dot(dz1)
                db1 = np.sum(dz1, axis=0)

                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            self.step_count += 1
            if self.step_count % self.target_update == 0:
                # copy to target
                if hasattr(self, 'W3'):
                    self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
                    self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()
                    self.tW3 = self.W3.copy(); self.tb3 = self.b3.copy()
                else:
                    self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
                    self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()

        def end_episode(self):
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        def save_checkpoint(self, path):
            """Save numpy-based network weights and epsilon to a .npz file."""
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            to_save = {'epsilon': self.epsilon}
            # save weights depending on architecture
            if hasattr(self, 'W3'):
                to_save.update({
                    'W1': self.W1, 'b1': self.b1,
                    'W2': self.W2, 'b2': self.b2,
                    'W3': self.W3, 'b3': self.b3,
                    'tW1': self.tW1, 'tb1': self.tb1,
                    'tW2': self.tW2, 'tb2': self.tb2,
                    'tW3': self.tW3, 'tb3': self.tb3,
                })
            else:
                to_save.update({
                    'W1': self.W1, 'b1': self.b1,
                    'W2': self.W2, 'b2': self.b2,
                    'tW1': self.tW1, 'tb1': self.tb1,
                    'tW2': self.tW2, 'tb2': self.tb2,
                })
            np.savez(path, **to_save)

        def load_checkpoint(self, path):
            """Load numpy-based network weights and epsilon from a .npz file."""
            data = np.load(path, allow_pickle=True)
            self.epsilon = float(data.get('epsilon', self.epsilon))
            if 'W3' in data:
                self.W1 = data['W1'].astype(np.float32)
                self.b1 = data['b1'].astype(np.float32)
                self.W2 = data['W2'].astype(np.float32)
                self.b2 = data['b2'].astype(np.float32)
                self.W3 = data['W3'].astype(np.float32)
                self.b3 = data['b3'].astype(np.float32)
                self.tW1 = data.get('tW1', self.W1).astype(np.float32)
                self.tb1 = data.get('tb1', self.b1).astype(np.float32)
                self.tW2 = data.get('tW2', self.W2).astype(np.float32)
                self.tb2 = data.get('tb2', self.b2).astype(np.float32)
                self.tW3 = data.get('tW3', self.W3).astype(np.float32)
                self.tb3 = data.get('tb3', self.b3).astype(np.float32)
            else:
                self.W1 = data['W1'].astype(np.float32)
                self.b1 = data['b1'].astype(np.float32)
                self.W2 = data['W2'].astype(np.float32)
                self.b2 = data['b2'].astype(np.float32)
                self.tW1 = data.get('tW1', self.W1).astype(np.float32)
                self.tb1 = data.get('tb1', self.b1).astype(np.float32)
                self.tW2 = data.get('tW2', self.W2).astype(np.float32)
                self.tb2 = data.get('tb2', self.b2).astype(np.float32)


    # Provide DQNSmall/Medium/Large wrappers
    class DQNSmall(NumpyDQNAgent):
        def __init__(self):
            super().__init__(hidden_sizes=(64,), lr=1e-3, batch_size=32, buffer_size=5000, target_update=20)


    class DQNMedium(NumpyDQNAgent):
        def __init__(self):
            super().__init__(hidden_sizes=(128, 64), lr=5e-4, batch_size=64, buffer_size=10000, target_update=10)


    class DQNLarge(NumpyDQNAgent):
        def __init__(self):
            super().__init__(hidden_sizes=(256, 128), lr=3e-4, batch_size=128, buffer_size=20000, target_update=5)