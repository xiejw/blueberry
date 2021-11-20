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

// put a new value into the board.
//
// Default value is 0 in states. Flag controls overwrite behavior.
error_t
boardSet(struct board_t *p, int row, int col, int v, int flag)
{
        // unsupported yet.
        assert(row == -1);
        assert(flag == 0);

        // find the first bottom row which is not filled yet.
        const int num_col = p->cols;
        for (int r = p->rows - 1; r >= 0; r--) {
                size_t offset = r * num_col + col;
                if (p->states[offset] == 0) {
                        // drop and return
                        p->states[offset] = v;
                        return OK;
                }
        }
        return errNew("the col: %d is full.", col);
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

        int u, v, k;

        // algorithrm: we do one pass scanning. We only find the longest 4
        // straight line for the direction we are sure it is needed.

#define ON_BOARD(r, c) (((r) >= 0 && (r) < rows) && ((c) >= 0 && (c) < cols))

        for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                        // error ignored as it is not possible to go wrong.
                        boardGet(b, r, c, &v);
                        if (v == PLAYER_NA) continue;

                        // only scan down
                        if (ON_BOARD(r - 1, c) &&
                            (boardGet(b, r - 1, c, &u), u == v)) {
                                // we can skip as it is examined already, when
                                // we deal with row r-1.
                                continue;
                        }

                        for (k = 1; k < num_to_win; k++) {
                                if (ON_BOARD(r + k, c) &&
                                    (boardGet(b, r + k, c, &u), u == v)) {
                                        continue;
                                }
                                break;
                        }
                        if (k == num_to_win) {
                                return v;
                        }
                }
        }

#undef ON_BOARD

        return PLAYER_NA;
}

// -----------------------------------------------------------------------------
// colors
// -----------------------------------------------------------------------------

#define COLOR_WINNER 1

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
}

// finialize ncurses scr
static void
finalizeScr()
{
        endwin();
}

// -----------------------------------------------------------------------------
// helpers.
// -----------------------------------------------------------------------------

int
main()
{
        // a standard 6x7 board for connect 4.
        struct board_t *b = boardNew(6, 7, 4, 1);

        const int     row_margin = 5;             // top margin for board.
        const int     col_margin = 15;            // left margin for board.
        int           pos        = 3;             // current placement column.
        enum player_t color      = PLAYER_BLACK;  // color for next stone.
        int           winner     = PLAYER_NA;

        error_t err;
        int     ch;  // input for getch().

        initScr();

        while (1) {
                int cur_row = 0;
                int v;

                // print instructions.
                mvprintw(cur_row++, 0,
                         "Use <- or -> to select column and space to place new "
                         "stone (q to quit).");

                if (winner != PLAYER_NA) {
                        // as here will be sufficient margin to plot board. we
                        // will not increase cur_row here.
                        assert(row_margin > 0);

                        attron(COLOR_PAIR(COLOR_WINNER));
                        mvprintw(cur_row, 0,
                                 " winner is: %d. press any key to quit",
                                 winner);
                        attroff(COLOR_PAIR(COLOR_WINNER));
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

                                if (v == 0) {
                                        printw("   |");
                                } else if (v > 0) {
                                        printw(" x |");
                                } else {
                                        printw(" o |");
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
                        if (c == pos) {
                                printw(" ^  ");
                        } else {
                                printw("    ");
                        }
                }
                mvprintw(cur_row++, col_margin, " ");
                for (int c = 0; c < b->cols; c++) {
                        if (c == pos) {
                                printw(" |  ");
                        } else {
                                printw("    ");
                        }
                }
                refresh();

                // keystroke events.

                ch = getch();

                if (winner != PLAYER_NA) {
                        ch = 'q';  // quit
                }

                switch (ch) {
                case CTRL('c'):
                case 'q':
                        refresh();
                        goto exit;
                case KEY_LEFT:
                        pos--;
                        if (pos < 0) {
                                pos = b->cols - 1;
                        }
                        break;
                case KEY_RIGHT:
                        pos++;
                        if (pos >= b->cols) {
                                pos = 0;
                        }
                        break;
                case ' ':
                        err = boardSet(b, -1, pos, color, 0);
                        if (OK != err) {
                                goto exit;
                        }
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
