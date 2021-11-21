#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <ncurses.h>
#undef OK  // conflict with eva, same value.

// eva
#include <base/error.h>

// -----------------------------------------------------------------------------
// data structures.
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
        int mode;  // ORed value of 1 (select col) 2 (select row)

        // internal
        int states[];
};

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
static int
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

// -----------------------------------------------------------------------------
// colors
// -----------------------------------------------------------------------------

#define COLOR_WINNER     1
#define COLOR_ERROR      2
#define COLOR_PREV_STONE 3

// -----------------------------------------------------------------------------
// error messages
// -----------------------------------------------------------------------------

#define ERR_MSG_COL_FULL "col is full, try again."

// -----------------------------------------------------------------------------
// helpers.
// -----------------------------------------------------------------------------

#define CTRL(c) ((c)&037)

// init ncurses scr
static void
initScr()
{
        initscr();             // start curses mode
        raw();                 // line buffering disabled
        keypad(stdscr, TRUE);  // get F1, F2 etc..
        noecho();              // don't echo() while we do getch
        curs_set(0);           // sets the cursor state to invisible

        start_color();
        init_pair(COLOR_WINNER, COLOR_BLACK, COLOR_GREEN);
        init_pair(COLOR_ERROR, COLOR_BLACK, COLOR_RED);
        init_pair(COLOR_PREV_STONE, COLOR_BLACK, COLOR_WHITE);
}

// finialize ncurses scr
static void
finalizeScr()
{
        endwin();
}

// -----------------------------------------------------------------------------
// main.
// -----------------------------------------------------------------------------

int
main()
{
        // a standard 6x7 board for connect 4.
        struct board_t *b = boardNew(6, 7, 4, 1);

        const int row_margin = 5;   // top margin for board.
        const int col_margin = 15;  // left margin for board.

        // non-local vars. used across moves.
        int           prev_row = -1;
        int           prev_col = -1;
        int           col      = 3;             // current placement column.
        enum player_t color    = PLAYER_BLACK;  // color for next stone.
        int           winner   = PLAYER_NA;
        char         *err_msg  = NULL;

        // local vars. used in small context.
        error_t err;
        int     ch;   // input for getch().
        int     row;  // track the current row to put, deduced by col.

        initScr();

        while (1) {
                int cur_row = 0;
                int v;

                clear();

                // print instructions.
                mvprintw(cur_row++, 0,
                         "Use <- or -> to select column and space to place new "
                         "stone (q to quit).");

                // for all msgs.
                //
                // four possible types (exclusive)
                // - winner
                // - error message
                // - info message
                // - (default) absent
                //
                // note: It will have sufficient margin to plot board. We will
                // not increase cur_row here.
                {
                        if (winner != PLAYER_NA) {
                                assert(err_msg == NULL);
                                assert(row_margin > 0);

                                attron(COLOR_PAIR(COLOR_WINNER));
                                mvprintw(cur_row, 0,
                                         " winner is: %d. press any key twice "
                                         "to quit",
                                         winner);
                                attroff(COLOR_PAIR(COLOR_WINNER));
                        }

                        if (err_msg != NULL) {
                                assert(winner == PLAYER_NA);
                                attron(COLOR_PAIR(COLOR_ERROR));
                                mvprintw(cur_row, 0, " error: %s", err_msg);
                                attroff(COLOR_PAIR(COLOR_ERROR));
                                err_msg = NULL;
                        }
                }

                // have some blank lines.
                cur_row += row_margin;

                // print the board.
                for (int r = 0; r < b->rows; r++) {
                        if (r == 0) {
                                // print the header
                                mvprintw(cur_row, col_margin, "+");
                                for (int c = 0; c < b->cols; c++) {
                                        printw("---+");
                                }
                                cur_row++;
                        }

                        mvprintw(cur_row++, col_margin, "|");
                        for (int c = 0; c < b->cols; c++) {
                                err = boardGet(b, r, c, &v);
                                if (err) {
                                        goto exit;
                                }

                                int set_color = prev_row == r && prev_col == c;

                                if (set_color) {
                                        printw(" ");

                                        attron(COLOR_PAIR(COLOR_PREV_STONE));
                                        assert(v != 0);
                                        if (v == PLAYER_BLACK) {
                                                printw("x");
                                        } else {
                                                assert(v == PLAYER_WHITE);
                                                printw("o");
                                        }
                                        attroff(COLOR_PAIR(COLOR_PREV_STONE));

                                        printw(" |");
                                } else {
                                        if (v == 0) {
                                                printw("   |");
                                        } else if (v == PLAYER_BLACK) {
                                                printw(" x |");
                                        } else {
                                                assert(v == PLAYER_WHITE);
                                                printw(" o |");
                                        }
                                }
                        }
                        mvprintw(cur_row++, col_margin, "+");
                        for (int c = 0; c < b->cols; c++) {
                                printw("---+");
                        }
                }

                // plot cursor point.
                mvprintw(cur_row++, col_margin, " ");
                for (int c = 0; c < b->cols; c++) {
                        if (c == col) {
                                printw(" ^  ");
                        } else {
                                printw("    ");
                        }
                }
                mvprintw(cur_row++, col_margin, " ");
                for (int c = 0; c < b->cols; c++) {
                        if (c == col) {
                                printw(" |  ");
                        } else {
                                printw("    ");
                        }
                }
                refresh();

                // keystroke events.

                ch = getch();

                if (winner != PLAYER_NA) {
                        getch();   // get another key to avoid accident.
                        ch = 'q';  // quit
                }

                switch (ch) {
                case CTRL('c'):
                case 'q':
                        refresh();
                        goto exit;
                case KEY_LEFT:
                        col--;
                        if (col < 0) {
                                col = b->cols - 1;
                        }
                        break;
                case KEY_RIGHT:
                        col++;
                        if (col >= b->cols) {
                                col = 0;
                        }
                        break;
                case ' ':
                        row = boardRowForCol(b, col);
                        if (row == -1) {
                                // will try again.
                                err_msg = ERR_MSG_COL_FULL;
                                break;
                        }
                        err = boardSet(b, row, col, color, 0);
                        if (OK != err) {
                                goto exit;
                        }

                        prev_row = row;
                        prev_col = col;

                        color =
                            color == PLAYER_BLACK ? PLAYER_WHITE : PLAYER_BLACK;
                        winner = boardWinner(b);
                        break;
                default:;
                }
        }

        // exit routing.
exit:
        finalizeScr();
        boardFree(b);

        if (err) {
                errDump("unexpected error.");
        } else {
                printf("bye!\n");
        }

        return 0;
}
