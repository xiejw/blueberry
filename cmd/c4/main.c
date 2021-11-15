#include <stdio.h>

#include <ncurses.h>

struct board_t {
        int rows;
        int cols;
        int mode;  // OR 1 (select col) 2 (select row)
};

// init ncurses scr
static void
initScr()
{
        initscr();             // start curses mode
        raw();                 // line buffering disabled
        keypad(stdscr, TRUE);  // get F1, F2 etc..
        noecho();              // don't echo() while we do getch
}

int
main()
{
        int ch;
        int pos = 3;  // current position.
        int cur_row;

        initScr();

        struct board_t b = {.rows = 6, .cols = 7, .mode = 1};

        while (1) {
                cur_row = 0;

                for (int r = 0; r < b.rows; r++) {
                        if (cur_row == 0) {
                                // print the header
                                mvprintw(cur_row, 0, "+");
                                for (int c = 0; c < b.cols; c++) {
                                        printw("---+");
                                }
                                cur_row++;
                        }

                        mvprintw(cur_row++, 0, "|");
                        for (int c = 0; c < b.cols; c++) {
                                printw("   |");
                        }
                        mvprintw(cur_row++, 0, "+");
                        for (int c = 0; c < b.cols; c++) {
                                printw("---+");
                        }
                }

                // plot cursor point.
                mvprintw(cur_row++, 0, " ");
                for (int c = 0; c < b.cols; c++) {
                        if (c == pos) {
                                printw(" ^  ");
                        } else {
                                printw("    ");
                        }
                }
                mvprintw(cur_row++, 0, " ");
                for (int c = 0; c < b.cols; c++) {
                        if (c == pos) {
                                printw(" |  ");
                        } else {
                                printw("    ");
                        }
                }

                cur_row++;  // have a blank line.

                mvprintw(cur_row++, 0,
                         "Use <- or -> to select column and space to place new "
                         "stone (q to quit).\n");
                refresh();

                ch = getch(); /* If raw() hadn't been called
                               * we have to press enter before it
                               * gets to the program 		*/
                switch (ch) {
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
                default:
                        refresh(); /* Print it on to the real screen */
                }
        }

exit:
        // end ncurses mode.
        endwin();
        printf("bye!\n");

        return 0;
}
