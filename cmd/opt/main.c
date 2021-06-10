#include <stdio.h>

#include "opt/fn.h"

int
main()
{
        sds_t s = sdsEmpty();
        printf("Hello opt.\n");

        struct bb_fn_t* fn = bbFnNew();
        bbFnDump(fn, &s);

        printf("%s", s);

        bbFnFree(fn);
        sdsFree(s);
        return 0;
}
