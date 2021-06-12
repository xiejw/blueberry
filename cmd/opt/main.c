#include <stdio.h>

#include "opt/fn.h"

#define APPEND(opcode, z, x, y) \
        bbFnAppend(             \
            fn, &(struct oparg_t){.op = opcode, .dst = z, .t1 = x, .t2 = y});

int
main()
{
        sds_t s = sdsEmpty();
        printf("Hello opt.\n");

        struct bb_fn_t *fn = bbFnNew();

        vecPushBack(fn->inputs, 1);
        vecPushBack(fn->inputs, 2);

        vecPushBack(fn->outputs, 7);

        APPEND(OP_ADD, 3, 1, 2);
        APPEND(OP_MUL, 6, 1, 1);
        APPEND(OP_MUL, 7, 6, 1);
        APPEND(OP_ADD, 4, 1, 2);
        APPEND(OP_ADD, 5, 1, 2);

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
