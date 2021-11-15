#include <stdio.h>

#include <ncurses.h>

struct board_t {
        int rows;
        int cols;
        int mode; // OR 1 (select col) 2 (select row)
};

        // init ncurses scr
static void initScr() {
        initscr();			// start curses mode
	raw();				// line buffering disabled
	keypad(stdscr, TRUE);		// get F1, F2 etc..
	noecho();			// don't echo() while we do getch
}

int
main()
{
        int ch;
        int pos; // current position.

        initScr();

        struct board_t b = { .rows= 6, .cols = 7, .mode = 1};

        for (int r = 0; r < b.rows; r++) {
                if (r==0) {
                        // print the header
                                        printw("+");
                        for (int c = 0; c < b.cols; c++) {
                                        printw("---+");
                        }
                }

                mvprintw(1 + 2 * r, 0, "|");
                for (int c = 0; c < b.cols; c++) {
                                        printw("   |");
                }
                mvprintw(1 + 2 * r + 1, 0, "+");
                for (int c = 0; c < b.cols; c++) {
                                        printw("---+");
                }
        }

    	printw("Type any character to see it in bold\n");
	ch = getch();			/* If raw() hadn't been called
					 * we have to press enter before it
					 * gets to the program 		*/
	if(ch == KEY_UP)		/* Without keypad enabled this will */
		printw("F1 Key pressed");/*  not get to us either	*/
					/* Without noecho() some ugly escape
					 * charachters might have been printed
					 * on screen			*/
	else
	{	printw("The pressed key is ");
		attron(A_BOLD);
		printw("%c", ch);
		attroff(A_BOLD);
	}
	refresh();			/* Print it on to the real screen */
    	getch();			/* Wait for user input */

        // end ncurses mode.
	endwin();

        return 0;
}
