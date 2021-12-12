#ifndef BB_BOT_H_
#define BB_BOT_H_

// eva
#include <adt/sds.h>

// bb
#include "board.h"

// -----------------------------------------------------------------------------
// bots
// -----------------------------------------------------------------------------

typedef error_t (*bot_fn)(struct board_t *, void *data, int prev_r, int prev_c,
                          int *r, int *c);

struct bot_t {
        sds_t  name;              // owned
        sds_t  msg;               // owned
        bot_fn bot_fn;            // the bot fn.
        void  *data;              // private data
        void (*free_fn)(void *);  // free fn to call if not NULL;
};

extern void botFree(struct bot_t *b);

extern struct bot_t *botNewDeterministic(const char *name, const char *msg);
extern struct bot_t *botNewMCTS(const char *name, const char *msg);

#endif  // BB_BOT_H_
