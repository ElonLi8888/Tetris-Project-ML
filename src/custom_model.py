"""Student Tetris controller.

The agent examines every legal rotation and horizontal position for the
current piece. It simulates each move and keeps the highest-scoring board.
"""

from copy import deepcopy


class CUSTOM_AI_MODEL:
    # Classic one-piece Tetris heuristic weights.
    HEIGHT_WEIGHT = -0.510066
    LINES_WEIGHT = 0.760666
    HOLES_WEIGHT = -0.356630
    BUMPINESS_WEIGHT = -0.184483

    def get_best_move(self, board, piece, depth=1):
        best_score = float("-inf")
        best_x = 0
        best_piece = piece

        for rotated_piece in self._unique_rotations(piece):
            piece_width = len(rotated_piece.skirt)

            for x in range(board.width - piece_width + 1):
                y = board.drop_height(rotated_piece, x)

                # Reject placements outside the visible and hidden board rows.
                if any(
                    y + block_y < 0
                    or y + block_y >= board.height + 4
                    for _, block_y in rotated_piece.body
                ):
                    continue

                simulated = deepcopy(board)
                if simulated.place(x, y, rotated_piece) != 0:
                    continue

                cleared = simulated.clear_rows()
                score = self._score(simulated, cleared)

                if score > best_score:
                    best_score = score
                    best_x = x
                    best_piece = rotated_piece

        return best_x, best_piece

    @staticmethod
    def _unique_rotations(piece):
        rotations = []
        seen = set()
        current = piece

        for _ in range(4):
            shape = tuple(sorted(current.body))
            if shape not in seen:
                seen.add(shape)
                rotations.append(current)
            current = current.get_next_rotation()

        return rotations

    def _score(self, board, cleared):
        heights = board.heights
        aggregate_height = sum(heights)
        bumpiness = sum(
            abs(heights[column] - heights[column + 1])
            for column in range(board.width - 1)
        )

        holes = 0
        for column, height in enumerate(heights):
            for row in range(height):
                if not board.board[row][column]:
                    holes += 1

        return (
            self.HEIGHT_WEIGHT * aggregate_height
            + self.LINES_WEIGHT * cleared
            + self.HOLES_WEIGHT * holes
            + self.BUMPINESS_WEIGHT * bumpiness
        )
