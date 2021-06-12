#include <stdio.h>

#include "opt/fn.h"

int
main()
{
        sds_t s = sdsEmpty();
        printf("Hello opt.\n");

        struct bb_fn_t* fn = bbFnNew();

        vecPushBack(fn->inputs, 1);
        vecPushBack(fn->inputs, 2);

        vecPushBack(fn->outputs, 3);

        bbFnAppend(fn,
                   &(struct oparg_t){.op = OP_ADD, .dst = 3, .t1 = 1, .t2 = 2});
        bbFnAppend(fn,
                   &(struct oparg_t){.op = OP_ADD, .dst = 4, .t1 = 1, .t2 = 2});
        bbFnAppend(fn,
                   &(struct oparg_t){.op = OP_ADD, .dst = 5, .t1 = 4, .t2 = 2});

        bbFnDump(fn, &s);
        printf("%s", s);

        int changed;
        int debug = 1;
        if (runDCEPass(fn, NULL, debug, &changed)) {
                errFatalAndExit1("something wrong.");
        }

        sdsClear(s);
        bbFnDump(fn, &s);
        printf("%s", s);

        bbFnFree(fn);
        sdsFree(s);
        return 0;
}
