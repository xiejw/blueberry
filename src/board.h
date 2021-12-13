#ifndef BB_BOARD_H_
#define BB_BOARD_H_

// eva
#include <base/error.h>

// -----------------------------------------------------------------------------
// Board and Player Data structures.
// -----------------------------------------------------------------------------

enum player_t {
        PLAYER_NA    = 0,    // use zero value.
        PLAYER_BLACK = 1,    // black stone/player.
        PLAYER_WHITE = -1,   // white stone/player.
        PLAYER_TIE   = 999,  // only used to decide winner.
};

struct board_t {
        // public
        int rows;
        int cols;
        int num_to_win;
        int mode;  // OR-ed value of 1 (select col) 2 (select row)

        // internal
        int states[];
};

// -----------------------------------------------------------------------------
// prototypes
// -----------------------------------------------------------------------------
extern struct board_t *boardNew(int rows, int cols, int num_to_win, int mode);
extern void            boardFree(struct board_t *p);

// Set and get the board position with value 'v'
extern error_t boardSet(struct board_t *b, int row, int col, int v, int flag);
extern error_t boardGet(struct board_t *p, int row, int col, int *v);

// Returns the row index when trying to place a stone in column 'col'; -1 on
// error.
extern int boardRowForCol(struct board_t *b, int col);

// Determines the current winner for board 'b'.
extern enum player_t boardWinner(struct board_t *b);

#endif  // BB_BOARD_H_
