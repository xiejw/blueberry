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

        struct board_t *p =
            calloc(1, sizeof(struct board_t *) + c * sizeof(int));
        p->rows = rows;
        p->cols = cols;
        p->mode = mode;

        return NULL;
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
        // TODO fill
        return OK;
}

// Get a new value from board and fill into `v`.
error_t
boardGet(struct board_t *p, int row, int col, int* v)
{
        size_t offset = row * p->cols + col;
        *v =  p->states[offset];
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
        struct board_t b = {.rows = 6, .cols = 7, .mode = 1};

        const int row_margin = 5;   // top margin for board.
        const int col_margin = 15;  // left margin for board.
        int       pos        = 3;   // current placement position (as column).

        int ch;  // input for getch().

        initScr();

        while (1) {
                int cur_row = 0;

                // print instructions.
                mvprintw(cur_row++, 0,
                         "Use <- or -> to select column and space to place new "
                         "stone (q to quit).\n");
                // have some blank lines.
                cur_row += row_margin;

                // print the board.
                for (int r = 0; r < b.rows; r++) {
                        if (r == 0) {
                                // print the header
                                mvprintw(cur_row, col_margin, "+");
                                for (int c = 0; c < b.cols; c++) {
                                        printw("---+");
                                }
                                cur_row++;
                        }

                        mvprintw(cur_row++, col_margin, "|");
                        for (int c = 0; c < b.cols; c++) {
                                printw("   |");
                        }
                        mvprintw(cur_row++, col_margin, "+");
                        for (int c = 0; c < b.cols; c++) {
                                printw("---+");
                        }
                }

                // plot cursor point.
                mvprintw(cur_row++, col_margin, " ");
                for (int c = 0; c < b.cols; c++) {
                        if (c == pos) {
                                printw(" ^  ");
                        } else {
                                printw("    ");
                        }
                }
                mvprintw(cur_row++, col_margin, " ");
                for (int c = 0; c < b.cols; c++) {
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
                                pos = b.cols - 1;
                        }
                        break;
                case KEY_RIGHT:
                        pos++;
                        if (pos >= b.cols) {
                                pos = 0;
                        }
                        break;
                default:;
                }
        }

        // exit routing.
exit:
        finalizeScr();

        printf("bye!\n");

        return 0;
}
