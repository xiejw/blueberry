#include "bot.h"

#include <unistd.h>  // sleep

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

void
botFree(struct bot_t *b)
{
        if (b == NULL) return;

        if (b->free_fn != NULL) {
                b->free_fn(b);
                return;
        }
        sdsFree(b->name);
        sdsFree(b->msg);
        free(b->data);
        free(b);
}
