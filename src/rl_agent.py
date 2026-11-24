from copy import deepcopy
from pathlib import Path
import pickle
import random

import numpy as np


MODEL_PATH = Path(__file__).with_name("tetris_rl_model.pkl")


def clone_board(board):
    copied = deepcopy(board)
    return copied


def board_features(board, lines_cleared=0):
    """Return small, normalized measurements of an after-placement board."""
    heights = np.asarray(board.heights, dtype=float)
    holes = 0
    for x, height in enumerate(board.heights):
        holes += sum(not board.board[y][x] for y in range(height))

    aggregate_height = heights.sum()
    bumpiness = np.abs(np.diff(heights)).sum()
    maximum_height = heights.max() if len(heights) else 0

    return np.asarray(
        [
            1.0,
            lines_cleared / 4.0,
            aggregate_height / 240.0,
            holes / 240.0,
            bumpiness / 216.0,
            maximum_height / 24.0,
        ],
        dtype=float,
    )


def unique_rotations(piece):
    rotations = []
    seen = set()
    current = piece
    for _ in range(4):
        signature = tuple(sorted(current.body))
        if signature not in seen:
            seen.add(signature)
            rotations.append(current)
        current = current.get_next_rotation()
    return rotations


def legal_afterstates(board, piece):
    """Generate every legal rotation/column and its resulting board."""
    candidates = []
    for rotated in unique_rotations(piece):
        piece_width = len(rotated.skirt)
        for x in range(board.width - piece_width + 1):
            try:
                y = board.drop_height(rotated, x)
            except (IndexError, ValueError):
                continue

            next_board = clone_board(board)
            result = next_board.place(x, y, rotated)
            if isinstance(result, Exception):
                continue

            lines = next_board.clear_rows()
            done = next_board.top_filled()
            features = board_features(next_board, lines)
            candidates.append(
                {
                    "x": x,
                    "piece": rotated,
                    "board": next_board,
                    "lines": lines,
                    "done": done,
                    "features": features,
                }
            )
    return candidates


class ReinforcementLearningAI:
    """A linear afterstate value agent trained with Q-learning self-play."""

    def __init__(self, model_path=MODEL_PATH, weights=None):
        self.model_path = Path(model_path)
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
        else:
            self.weights = self.load_weights()

    def load_weights(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {self.model_path}. "
                "Run: python src/train_rl.py"
            )
        with self.model_path.open("rb") as file:
            saved = pickle.load(file)
        return np.asarray(saved["weights"], dtype=float)

    def value(self, features):
        return float(np.dot(self.weights, features))

    def choose_candidate(self, board, piece, epsilon=0.0):
        candidates = legal_afterstates(board, piece)
        if not candidates:
            return None
        if random.random() < epsilon:
            return random.choice(candidates)
        return max(candidates, key=lambda candidate: self.value(candidate["features"]))

    def get_best_move(self, board, piece, depth=1):
        candidate = self.choose_candidate(board, piece, epsilon=0.0)
        if candidate is None:
            return 0, piece
        return candidate["x"], candidate["piece"]

