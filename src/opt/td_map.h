// mlvm
#include "vm_internal.h"  //  MAX_TENSOR_COUNT

// -----------------------------------------------------------------------------
// Map Helpers.
// -----------------------------------------------------------------------------

// a fast map specific for the tensor descriptor. This structure assumes that
// the input must SSA-like and all tensor descriptors are contiguous.
struct td_map_t {
        int cap;
        vec_t(void*) data;
};

struct td_map_t*
bbTdMapNew()
{
        int              cap = MLVM_MAX_TENSOR_COUNT;
        struct td_map_t* p   = malloc(sizeof(struct td_map_t));
        p->cap               = cap;
        p->data              = NULL;
        vecReserve(p->data, cap);
        memset(p->data, 0, cap * sizeof(void*));  // NULL all slots.
        return p;
}

void
bbTdMapFree(struct td_map_t* p)
{
        if (p == NULL) return;
        vecFree(p->data);
        free(p);
}

error_t
bbTdMapFind(struct td_map_t* map, int td, void** data)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");
        *data = map->data[td];
        return OK;
}

error_t
bbTdMapSet(struct td_map_t* map, int td, void* v)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");

        void** p = &map->data[td];

        if (*p != NULL) return errNew("already have value.");

        *p = v;
        return OK;
}
