import numpy as np

class OwareBoard:
    def __init__(self, variant="standard"):
        """
        OwareBoard supports simple rule variants.

        Variants supported:
        - "standard": 4 seeds per pit, chain captures allowed (default)
        - "sparse": 2 seeds per pit, chain captures allowed
        - "dense": 6 seeds per pit, chain captures allowed
        - "no_chain": 4 seeds per pit, but only capture the last pit (no backward chaining)
        """
        self.variant = variant
        if variant == "sparse":
            self.initial_seeds = 2
            self.capture_chain = True
        elif variant == "dense":
            self.initial_seeds = 6
            self.capture_chain = True
        elif variant == "no_chain":
            self.initial_seeds = 4
            self.capture_chain = False
        else:
            self.initial_seeds = 4
            self.capture_chain = True

        # total seeds used for endgame threshold
        self.total_seeds = self.initial_seeds * 12
        self.reset()

    def reset(self):
        self.board = np.array([self.initial_seeds] * 12, dtype=int)
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
        # check if the last pit is on the opponent's side
        opponent_side = (self.current_player == 0 and 6 <= pit <= 11) or (
            self.current_player == 1 and 0 <= pit <= 5
        )
        if not opponent_side:
            return

        # capture rule: capture if pit has 2 or 3 seeds
        if self.board[pit] in (2, 3):
            # capture this pit
            self.scores[self.current_player] += int(self.board[pit])
            self.board[pit] = 0
            if not self.capture_chain:
                return
            # if capture_chain is True, continue backward as long as pits are 2 or 3
            pit = (pit - 1) % 12
            opponent_side = (self.current_player == 0 and 6 <= pit <= 11) or (
                self.current_player == 1 and 0 <= pit <= 5
            )
            while opponent_side and self.board[pit] in (2, 3):
                self.scores[self.current_player] += int(self.board[pit])
                self.board[pit] = 0
                pit = (pit - 1) % 12
                opponent_side = (self.current_player == 0 and 6 <= pit <= 11) or (
                    self.current_player == 1 and 0 <= pit <= 5
                )

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