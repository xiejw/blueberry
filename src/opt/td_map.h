#ifndef BB_OPT_TD_MAP_H_
#define BB_OPT_TD_MAP_H_

#include "adt/vec.h"

// -----------------------------------------------------------------------------
// Map Helpers.
// -----------------------------------------------------------------------------

// a fast map specific for the tensor descriptor. This structure assumes that
// the input must SSA-like and all tensor descriptors are contiguous.
struct td_map_t {
        int cap;
        vec_t(void*) data;
};

struct td_map_t* bbTdMapNew();
void             bbTdMapFree(struct td_map_t* p);
error_t          bbTdMapFind(struct td_map_t* map, int td, void** data);
error_t          bbTdMapSet(struct td_map_t* map, int td, void* v);

#endif
