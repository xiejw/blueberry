#include <stdio.h>
#include <stdlib.h>

// eva
#include <base/error.h>

// bb
#include <board.h>
#include <bot.h>
#include <runner.h>

// -----------------------------------------------------------------------------
// main.
// -----------------------------------------------------------------------------

int
main()
{
        // a standard 6x7 board for connect 4.
        // struct board_t *b = boardNew(6, 7, 4, 1);

        // TODO: recover the 6x7 board.
        // testing purpose 4x5 x 3, 1
        struct board_t *b = boardNew(4, 5, 3, 1);

        // bots (NULL is human)

        struct bot_t *bot_black = botNewRandom("black", "random", /*seed=*/23);
        struct bot_t *bot_white = botNewDeterministic(
            "white", "deterministic bot is playing for few seconds...");

        // runner starts.
        error_t err = runner(b, bot_black, bot_white, /*final_winner=*/NULL);

        // exit routing.
        boardFree(b);

        botFree(bot_black);
        botFree(bot_white);

        if (err) {
                errDump("unexpected error.");
        } else {
                printf("bye!\n");
        }

        return 0;
}
