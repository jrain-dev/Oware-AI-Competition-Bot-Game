import numpy as np

class OwareBoard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.array([4] * 12, dtype=int)
        self.scores = [0, 0]
        self.current_player = 0
        self.game_over = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return tuple(self.board)

    def get_valid_moves(self, player):
        if player == 0:
            player_pits = self.board[0:6]
            offset = 0
        else:
            player_pits = self.board[6:12]
            offset = 6
        return [i + offset for i, seeds in enumerate(player_pits) if seeds > 0]

    def apply_move(self, move):
        if self.game_over:
            return 0, self.get_state(), True
        seeds = int(self.board[move])
        self.board[move] = 0
        pit = move
        while seeds > 0:
            pit = (pit + 1) % 12
            if pit == move:
                continue
            self.board[pit] += 1
            seeds -= 1
        self._handle_capture(pit)
        self.current_player = 1 - self.current_player
        self._check_game_over()
        reward = 0
        if self.game_over:
            if self.winner == 0:
                reward = 1
            elif self.winner == 1:
                reward = -1
        return reward, self.get_state(), self.game_over

    def _handle_capture(self, last_pit):
        pit = last_pit
        opponent_side = (self.current_player == 0 and 6 <= pit <= 11) or (
            self.current_player == 1 and 0 <= pit <= 5
        )
        while opponent_side:
            if self.board[pit] in (2, 3):
                self.scores[self.current_player] += int(self.board[pit])
                self.board[pit] = 0
                pit = (pit - 1) % 12
                opponent_side = (self.current_player == 0 and 6 <= pit <= 11) or (
                    self.current_player == 1 and 0 <= pit <= 5
                )
            else:
                break

    def _check_game_over(self):
        if self.scores[0] > 24:
            self.game_over = True
            self.winner = 0
            return
        if self.scores[1] > 24:
            self.game_over = True
            self.winner = 1
            return
        if not self.get_valid_moves(self.current_player):
            self.game_over = True
            self.scores[1 - self.current_player] += int(np.sum(self.board))
            if self.scores[0] > self.scores[1]:
                self.winner = 0
            elif self.scores[1] > self.scores[0]:
                self.winner = 1
            else:
                self.winner = -1

    def __str__(self):
        p1 = " ".join(map(str, self.board[11:5:-1]))
        p0 = " ".join(map(str, self.board[0:6]))
        return f"P1 Score: {self.scores[1]}\n Pits: {p1}\n       {p0}\nP0 Score: {self.scores[0]}"