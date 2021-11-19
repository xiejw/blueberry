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

struct board_t {
        // public
        int rows;
        int cols;
        int mode;  // OR 1 (select col) 2 (select row)

        // internal
        int states[];
};

struct board_t *
boardNew(int rows, int cols, int mode)
{
        size_t c = rows * cols;
        assert(c > 0);

        struct board_t *p = calloc(1, sizeof(struct board_t) + c * sizeof(int));
        p->rows           = rows;
        p->cols           = cols;
        p->mode           = mode;

        return p;
}

void
boardFree(struct board_t *p)
{
        free(p);
}

// Put a new value into the board.
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

// Get a new value from board and fill into `v`.
error_t
boardGet(struct board_t *p, int row, int col, int *v)
{
        size_t offset = row * p->cols + col;
        *v            = p->states[offset];
        return OK;
}

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
        struct board_t *b = boardNew(6, 7, 1);

        const int row_margin = 5;   // top margin for board.
        const int col_margin = 15;  // left margin for board.
        int       pos        = 3;   // current placement position (as column).
        int       color      = 1;   // color for next stone.

        error_t err;
        int     ch;  // input for getch().

        initScr();

        while (1) {
                int cur_row = 0;
                int v;

                // print instructions.
                mvprintw(cur_row++, 0,
                         "Use <- or -> to select column and space to place new "
                         "stone (q to quit).\n");
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
                        err = boardSet(b, -1, pos, /*v=*/color, 0);
                        if (OK != err) {
                                goto exit;
                        }
                        color = color * -1;
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
