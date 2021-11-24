#include "runner.h"

#include <ncurses.h>

// eva
#include <base/error.h>

// -----------------------------------------------------------------------------
// colors
// -----------------------------------------------------------------------------

#define COLOR_WINNER     1
#define COLOR_ERROR      2
#define COLOR_PREV_STONE 3
#define COLOR_BOT        4

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
        init_pair(COLOR_BOT, COLOR_BLACK, COLOR_CYAN);
}

// finialize ncurses scr
static void
finalizeScr()
{
        endwin();
}

// -----------------------------------------------------------------------------
// runner.
// -----------------------------------------------------------------------------

error_t
runner(struct board_t *b, struct bot_t *bot_black, struct bot_t *bot_white,
       int *final_winner)
{
        // static configurations.
        const int row_margin = 5;   // top margin for board.
        const int col_margin = 15;  // left margin for board.

        // non-local vars. used across moves.
        int           prev_row = -1;
        int           prev_col = -1;
        int           col      = 3;             // current placement column.
        enum player_t color    = PLAYER_BLACK;  // color for next stone.
        int           winner   = PLAYER_NA;     // winner of the game.
        char         *err_msg  = NULL;          // recoverable errors.
        error_t       err      = OK;

        // local vars. used in small context.
        int ch;   // input for getch().
        int row;  // track the current row to put, deduced by col.

        initScr();

        while (1) {
                int cur_row = 0;
                int v;

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
                        // this is an optimization. we clear the line
                        // for error message, etc rather than clear the
                        // whole screen.
                        move(cur_row, 0);
                        clrtoeol();

                        if (winner != PLAYER_NA) {
                                assert(err_msg == NULL);
                                assert(row_margin > 0);

                                attron(COLOR_PAIR(COLOR_WINNER));
                                mvprintw(cur_row, 0,
                                         " winner is: %d. press any key twice "
                                         "to quit",
                                         winner);
                                attroff(COLOR_PAIR(COLOR_WINNER));
                        } else if (err_msg != NULL) {
                                assert(winner == PLAYER_NA);
                                attron(COLOR_PAIR(COLOR_ERROR));
                                mvprintw(cur_row, 0, " error: %s", err_msg);
                                attroff(COLOR_PAIR(COLOR_ERROR));
                                err_msg = NULL;
                        } else if (color == PLAYER_BLACK && bot_black != NULL) {
                                assert(winner == PLAYER_NA);
                                assert(err_msg == NULL);
                                attron(COLOR_PAIR(COLOR_BOT));
                                mvprintw(cur_row, 0, "%s: %s", bot_black->name,
                                         bot_black->msg);
                                attroff(COLOR_PAIR(COLOR_BOT));
                        } else if (color == PLAYER_WHITE && bot_white != NULL) {
                                assert(winner == PLAYER_NA);
                                assert(err_msg == NULL);
                                attron(COLOR_PAIR(COLOR_BOT));
                                mvprintw(cur_row, 0, "%s: %s", bot_white->name,
                                         bot_white->msg);
                                attroff(COLOR_PAIR(COLOR_BOT));
                        } else {
                                // no action.
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

                // handle player/bot logic now.

                struct bot_t *bot =
                    color == PLAYER_BLACK ? bot_black : bot_white;

                if (winner != PLAYER_NA) {
                        // we have a winner. give users some time to check the
                        // result and then quit.
                        getch();
                        getch();   // get another key to avoid accident.
                        ch = 'q';  // quit. fall through.
                } else if (bot != NULL) {
                        assert(winner == PLAYER_NA);
                        int r, c;  // dont pollute the pos for the UI.
                        err = bot->bot_fn(b, bot->data, prev_row, prev_col, &r,
                                          &c);
                        if (err) {
                                err = errEmitNote(
                                    "unexpected error during playing bot.");
                                goto exit;
                        }

                        err = boardSet(b, r, c, color, 0);
                        if (OK != err) {
                                err = errEmitNote(
                                    "unexpected error during placing stone for "
                                    "the bot.");
                                goto exit;
                        }

                        prev_row = r;
                        prev_col = c;

                        color =
                            color == PLAYER_BLACK ? PLAYER_WHITE : PLAYER_BLACK;
                        winner = boardWinner(b);

                        // we are done here. either, we found a winner and plot
                        // the winning move in next iteration and quit, or we
                        // start next iteration.
                        continue;

                } else {
                        // human, check keystroke.
                        ch = getch();
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
                                err = errEmitNote(
                                    "unexpected error during placing stone for "
                                    "the user.");
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

exit:
        finalizeScr();
        if (final_winner != NULL) {
                *final_winner = winner;
        }
        return err;
}
