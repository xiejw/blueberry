#ifndef BB_RUNNER_H_
#define BB_RUNNER_H_

// bb
#include "board.h"
#include "bot.h"

// -----------------------------------------------------------------------------
// runner.
// -----------------------------------------------------------------------------

// params:
//
//   - b: the board
//   - bot_black: the bot for black player. NULL-able, not owned.
//   - bot_white: the bot for white player. NULL-able, not owned.
//
//   - final_winner: if not NULL, set as enum player_t (NA means users cancel
//   the game).
//
// return value:
//   same as error_t.
error_t runner(struct board_t *b, struct bot_t *bot_black,
               struct bot_t *bot_white, int *final_winner);

#endif  // BB_RUNNER_H_
