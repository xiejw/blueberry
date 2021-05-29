#include <stdio.h>

// bb
#include "bb.h"

int
main()
{
        printf("hello bb.\n");
        struct vm_t* vm = bbVmInit();

        vmFree(vm);
        return OK;
}
