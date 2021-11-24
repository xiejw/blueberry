#include "board.h"

#include <assert.h>
#include <stdlib.h>

struct board_t *
boardNew(int rows, int cols, int num_to_win, int mode)
{
        size_t c = rows * cols;
        assert(c > 0);

        struct board_t *p = calloc(1, sizeof(struct board_t) + c * sizeof(int));
        p->rows           = rows;
        p->cols           = cols;
        p->num_to_win     = num_to_win;
        p->mode           = mode;

        return p;
}

void
boardFree(struct board_t *p)
{
        free(p);
}

// find the row to put the col or -1 if the col is full.
int
boardRowForCol(struct board_t *b, int col)
{
        // find the first bottom row which is not filled yet.
        const int num_col = b->cols;
        for (int r = b->rows - 1; r >= 0; r--) {
                size_t offset = r * num_col + col;
                if (b->states[offset] == PLAYER_NA) {
                        return r;
                }
        }
        return -1;
}

// put a new value into the board.
//
// Default value is 0 in states. Flag controls overwrite behavior.
error_t
boardSet(struct board_t *b, int row, int col, int v, int flag)
{
        // unsupported yet.
        assert(flag == 0);

        size_t offset     = row * b->cols + col;
        b->states[offset] = v;
        return OK;
}

// get a new value from board and fill into `v`.
error_t
boardGet(struct board_t *p, int row, int col, int *v)
{
        size_t offset = row * p->cols + col;
        *v            = p->states[offset];
        return OK;
}

enum player_t
boardWinner(struct board_t *b)
{
        const int rows       = b->rows;
        const int cols       = b->cols;
        const int num_to_win = b->num_to_win;

        int num_stones = 0;
        int u, v, k;

        // algorithrm: we do one pass scanning. We only find the longest 4
        // straight line for the direction we are sure it is needed.

#define ON_BOARD(r, c) (((r) >= 0 && (r) < rows) && ((c) >= 0 && (c) < cols))

        for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                        // error ignored as it is not possible to go wrong.
                        boardGet(b, r, c, &v);
                        if (v == PLAYER_NA) continue;

                        num_stones++;

                        // statically unroll to avoid unnecessary function calls

                        // only scan down
                        {
                                if (ON_BOARD(r - 1, c) &&
                                    (boardGet(b, r - 1, c, &u), u == v)) {
                                        // we can skip as it is examined
                                        // already, when we deal with row r-1.
                                        continue;
                                }

                                for (k = 1; k < num_to_win; k++) {
                                        if (ON_BOARD(r + k, c) &&
                                            (boardGet(b, r + k, c, &u),
                                             u == v)) {
                                                continue;
                                        }
                                        break;
                                }
                                if (k == num_to_win) {
                                        return v;
                                }
                        }

                        // only scan right
                        {
                                if (ON_BOARD(r, c - 1) &&
                                    (boardGet(b, r, c - 1, &u), u == v)) {
                                        // we can skip as it is examined
                                        // already, when we deal with col c-1.
                                        continue;
                                }

                                for (k = 1; k < num_to_win; k++) {
                                        if (ON_BOARD(r, c + k) &&
                                            (boardGet(b, r, c + k, &u),
                                             u == v)) {
                                                continue;
                                        }
                                        break;
                                }
                                if (k == num_to_win) {
                                        return v;
                                }
                        }

                        // only scan right down
                        {
                                if (ON_BOARD(r - 1, c - 1) &&
                                    (boardGet(b, r - 1, c - 1, &u), u == v)) {
                                        // we can skip as it is examined
                                        // already, when we deal with row r-1,
                                        // col c-1.
                                        continue;
                                }

                                for (k = 1; k < num_to_win; k++) {
                                        if (ON_BOARD(r + k, c + k) &&
                                            (boardGet(b, r + k, c + k, &u),
                                             u == v)) {
                                                continue;
                                        }
                                        break;
                                }
                                if (k == num_to_win) {
                                        return v;
                                }
                        }

                        // only scan right up
                        {
                                if (ON_BOARD(r + 1, c - 1) &&
                                    (boardGet(b, r + 1, c - 1, &u), u == v)) {
                                        // we can skip as it is examined
                                        // already, when we deal with row r+1,
                                        // col c-1.
                                        continue;
                                }

                                for (k = 1; k < num_to_win; k++) {
                                        if (ON_BOARD(r - k, c + k) &&
                                            (boardGet(b, r - k, c + k, &u),
                                             u == v)) {
                                                continue;
                                        }
                                        break;
                                }
                                if (k == num_to_win) {
                                        return v;
                                }
                        }
                }
        }

#undef ON_BOARD

        return num_stones == rows * cols ? PLAYER_TIE : PLAYER_NA;
}
