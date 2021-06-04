#include "opt/fn.h"

struct bb_fn_t*
bbFnNew()
{
        size_t          psize = sizeof(struct bb_fn_t);
        struct bb_fn_t* p     = malloc(psize);
        memset(p, 0, psize);
        return p;
}

void
bbFnFree(struct bb_fn_t* p)
{
        if (p == NULL) return;
        free(p);
}
