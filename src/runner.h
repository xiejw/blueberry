#ifndef BB_RUNNER_H_
#define BB_RUNNER_H_

// bb
#include "board.h"
#include "bot.h"

// -----------------------------------------------------------------------------
// Runner.
// -----------------------------------------------------------------------------

// Params:
//
//   - b: the board
//   - bot_black: the bot for black player. NULL-able, not owned.
//   - bot_white: the bot for white player. NULL-able, not owned.
//
//   - final_winner: if not NULL, set as enum player_t (NA means users cancel
//   the game).
//
// Return value:
//   same as error_t.
error_t runner(struct board_t *b, struct bot_t *bot_black,
               struct bot_t *bot_white, _out_ int *final_winner);

#endif  // BB_RUNNER_H_
