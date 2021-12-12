#include "bot.h"

#include <unistd.h>  // sleep

// eva
#include <rng/srng64.h>

// -----------------------------------------------------------------------------
// general public APis for all bots.
// -----------------------------------------------------------------------------

void
botFree(struct bot_t *b)
{
        if (b == NULL) return;

        if (b->free_fn != NULL) {
                // here the contract is: once free_fn is provided, it is
                // responsible not only for the data but also all other fields.
                //
                // this is useful to let free_fn to manage the name, msg
                // differently.
                b->free_fn(b);
                return;
        }

        sdsFree(b->name);
        sdsFree(b->msg);
        free(b->data);
        free(b);
}

// -----------------------------------------------------------------------------
// deterministic bot.
// -----------------------------------------------------------------------------

// always place a stone in the first legitimate col.
static error_t
bot_fn_deter(struct board_t *b, void *data, int prev_r, int prev_c, int *r,
             int *c)
{
        const int cols = b->cols;
        int       row;

        for (int col = 0; col < cols; col++) {
                row = boardRowForCol(b, col);
                if (row != -1) {
                        *r = row;
                        *c = col;
                        return OK;
                }
        }
        return errNew("board is full.");
}

static error_t
bot_fn_deter_sleep(struct board_t *b, void *data, int prev_r, int prev_c,
                   int *r, int *c)
{
        sleep(1);  // sleep for 1 sec to mimic a game.
        return bot_fn_deter(b, data, prev_r, prev_c, r, c);
}

struct bot_t *
botNewDeterministic(const char *name, const char *msg)
{
        struct bot_t *p = malloc(sizeof(*p));
        p->name         = sdsNew(name);
        p->msg          = sdsNew(msg);
        p->bot_fn       = bot_fn_deter_sleep;
        p->data         = NULL;
        p->free_fn      = NULL;
        return p;
}

// -----------------------------------------------------------------------------
// random bot.
// -----------------------------------------------------------------------------
static void
random_free_fn(void *bot_p)
{
        struct bot_t   *b = (struct bot_t *)bot_p;
        struct rng64_t *p = b->data;

        rng64Free(p);
        b->data    = NULL;
        b->free_fn = NULL;
        botFree(b);
}

// always place a stone in random place.
static error_t
bot_fn_random(struct board_t *b, void *data, int prev_r, int prev_c, int *r,
              int *c)
{
        const int       cols = b->cols;
        struct rng64_t *p    = data;

        int row;  // in loop usage.

        int total_tries = 0;

        while (total_tries++ <= 1000) {
                int col = rng64NextUint64(p) % cols;
                row     = boardRowForCol(b, col);
                if (row != -1) {
                        *r = row;
                        *c = col;
                        return OK;
                }
        }
        return errNew("reached max loop attemps. board is likely full.");
}

struct bot_t *
botNewRandom(const char *name, const char *msg, uint64_t seed)
{
        struct bot_t *p = malloc(sizeof(*p));
        p->name         = sdsNew(name);
        p->msg          = sdsNew(msg);
        p->bot_fn       = bot_fn_random;
        p->data         = srng64New(seed);
        p->free_fn      = random_free_fn;

        return p;
}

// -----------------------------------------------------------------------------
// monte carlo tree search (MTTS) bot.
// -----------------------------------------------------------------------------
struct bot_t *
botNewMCTS(const char *name, const char *msg, uint64_t seed)
{
        struct bot_t *p = malloc(sizeof(*p));
        p->name         = sdsNew(name);
        p->msg          = sdsNew(msg);
        p->bot_fn       = NULL;  // TODO: replace here.
        p->data         = NULL;
        p->free_fn      = NULL;

        return p;
}
