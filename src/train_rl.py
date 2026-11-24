from pathlib import Path
import argparse
import pickle
import random

import numpy as np

from board import Board
from piece import Piece
from rl_agent import MODEL_PATH, ReinforcementLearningAI, board_features, legal_afterstates


def reward_for_move(old_board, candidate):
    new_features = candidate["features"]
    aggregate_height = new_features[2] * 240.0
    holes = new_features[3] * 240.0
    bumpiness = new_features[4] * 216.0
    maximum_height = new_features[5] * 24.0

    # Reward good play directly. This shaped reward lets the agent learn from
    # every placement instead of receiving useful feedback only after losing.
    reward = 1.0
    reward += 20.0 * (candidate["lines"] ** 2)
    reward -= 0.35 * aggregate_height
    reward -= 2.5 * holes
    reward -= 0.25 * bumpiness
    reward -= 0.75 * maximum_height
    if candidate["done"]:
        reward -= 100.0
    return reward


def evaluate(weights, games=10, piece_limit=1000):
    scores = []
    rows = []
    agent = ReinforcementLearningAI(weights=weights)
    for _ in range(games):
        board = Board()
        pieces = 0
        cleared = 0
        while pieces < piece_limit and not board.top_filled():
            candidate = agent.choose_candidate(board, Piece())
            if candidate is None:
                break
            board = candidate["board"]
            pieces += 1
            cleared += candidate["lines"]
        scores.append(pieces)
        rows.append(cleared)
    return float(np.mean(scores)), float(np.mean(rows))


def train(episodes=200, piece_limit=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    # Six trainable values: bias, cleared lines, height, holes,
    # bumpiness, and maximum height. They begin untrained at zero.
    weights = np.zeros(6, dtype=float)
    agent = ReinforcementLearningAI(weights=weights)

    gamma = 0.95
    learning_rate = 0.01
    epsilon_start = 1.0
    epsilon_end = 0.03

    history = []
    for episode in range(1, episodes + 1):
        progress = (episode - 1) / max(1, episodes - 1)
        epsilon = epsilon_start * ((epsilon_end / epsilon_start) ** progress)

        board = Board()
        pieces = 0
        rows = 0

        while pieces < piece_limit and not board.top_filled():
            piece = Piece()
            candidates = legal_afterstates(board, piece)
            if not candidates:
                break

            if random.random() < epsilon:
                candidate = random.choice(candidates)
            else:
                candidate = max(
                    candidates,
                    key=lambda item: float(np.dot(weights, item["features"])),
                )

            reward = reward_for_move(board, candidate)
            prediction = float(np.dot(weights, candidate["features"]))

            if candidate["done"] or pieces + 1 >= piece_limit:
                target = reward
            else:
                next_piece = Piece()
                next_candidates = legal_afterstates(candidate["board"], next_piece)
                next_value = max(
                    (float(np.dot(weights, item["features"])) for item in next_candidates),
                    default=0.0,
                )
                target = reward + gamma * next_value

            td_error = np.clip(target - prediction, -10.0, 10.0)
            weights += learning_rate * td_error * candidate["features"]
            weights = np.clip(weights, -500.0, 500.0)

            board = candidate["board"]
            pieces += 1
            rows += candidate["lines"]

        history.append((pieces, rows))
        if episode == 1 or episode % 100 == 0:
            recent = history[-100:]
            avg_pieces = np.mean([item[0] for item in recent])
            avg_rows = np.mean([item[1] for item in recent])
            print(
                f"Episode {episode:4d}/{episodes} | epsilon={epsilon:.3f} | "
                f"average pieces={avg_pieces:.1f} | average rows={avg_rows:.1f}"
            )

    return weights, history


def main():
    parser = argparse.ArgumentParser(description="Train the Tetris RL agent by self-play.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--piece-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights, history = train(args.episodes, args.piece_limit, args.seed)
    average_pieces, average_rows = evaluate(weights)

    saved = {
        "weights": weights,
        "episodes": args.episodes,
        "piece_limit": args.piece_limit,
        "seed": args.seed,
        "training_history": history,
        "evaluation_average_pieces": average_pieces,
        "evaluation_average_rows": average_rows,
    }
    with Path(MODEL_PATH).open("wb") as file:
        pickle.dump(saved, file)

    print("\nTraining complete")
    print("Learned weights:", np.round(weights, 4))
    print(f"Evaluation average pieces: {average_pieces:.1f}")
    print(f"Evaluation average rows: {average_rows:.1f}")
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()

