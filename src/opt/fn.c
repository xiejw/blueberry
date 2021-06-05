#include "opt/fn.h"

// -----------------------------------------------------------------------------
// General map.
// -----------------------------------------------------------------------------

#include <limits.h>

#define DICT_HT_INITIAL_SIZE 128

struct dict_entry_t {
        void *key;
        union {
                void *   val;
                uint64_t u64;
                int64_t  s64;
                double   d;
        } v;
        struct dict_entry_t *next;
};

struct dict_ty_t {
        uint64_t (*hashFn)(const void *key);
        void *(*keyDup)(void *privdata, const void *key);
        void *(*valDup)(void *privdata, const void *obj);
        int (*keyCmp)(void *privdata, const void *key1, const void *key2);
        void (*keyFree)(void *privdata, void *key);
        void (*valFree)(void *privdata, void *obj);
};

struct dict_table_t {
        struct dict_entry_t **table;
        unsigned long         size;
        unsigned long         sizemask;
};

struct dict_t {
        struct dict_ty_t *  type;
        void *              privdata;
        struct dict_table_t ht;
};

static void
_dictReset(struct dict_table_t *ht)
{
        ht->table    = NULL;
        ht->size     = 0;
        ht->sizemask = 0;
}

error_t
_dictInit(struct dict_t *d, struct dict_ty_t *type, void *privDataPtr)
{
        _dictReset(&d->ht);
        d->type     = type;
        d->privdata = privDataPtr;
        return OK;
}

/* Create a new hash table */
struct dict_t *
dictNew(struct dict_ty_t *type, void *privDataPtr)
{
        struct dict_t *d = malloc(sizeof(*d));
        _dictInit(d, type, privDataPtr);
        return d;
}

/* Our hash table capability is a power of two */
static unsigned long
_dictNextPower(unsigned long size)
{
        unsigned long i = DICT_HT_INITIAL_SIZE;

        if (size >= LONG_MAX) return LONG_MAX + 1LU;
        while (1) {
                if (i >= size) return i;
                i *= 2;
        }
}

error_t
_dictExpand(struct dict_t *d, unsigned long size)
{
        struct dict_table_t n; /* the new hash table */
        unsigned long       realsize = _dictNextPower(size);

        if (realsize == d->ht.size) return OK;

        /* Allocate the new hash table and initialize all pointers to NULL */
        n.size     = realsize;
        n.sizemask = realsize - 1;
        n.table    = malloc(realsize * sizeof(struct dict_entry_t *));

        /* Is this the first initialization? If so it's not really a rehashing
         * we just set the first hash table so that it can accept keys. */
        d->ht = n;
        return OK;
}

/* return DICT_ERR if expand was not performed */
error_t
dictExpand(struct dict_t *d, unsigned long size)
{
        return _dictExpand(d, size);
}

// -----------------------------------------------------------------------------
// Map Helpers.
// -----------------------------------------------------------------------------

// a fast map specific for the tensor descriptor. This structure assumes that
// the input must SSA-like and all tensor descriptor is contiguous.
struct td_map_t {
        int cap;
        vec_t(void *) data;
};

struct td_map_t *
bbTdMap(int cap)
{
        struct td_map_t *p = malloc(sizeof(struct td_map_t));
        p->cap             = cap;
        p->data            = NULL;
        vecReserve(p->data, cap);
        memset(p->data, 0, cap * sizeof(void *));  // NULL all slots.
        return p;
}

error_t
bbTdMapFind(struct td_map_t *map, int td, void **data)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");
        *data = map->data[td];
        return OK;
}

#define BB_TD_MAP_OVERWRITE        0
#define BB_TD_MAP_DO_NOT_OVERWRITE 1

error_t
bbTdMapSet(struct td_map_t *map, int td, void *v, int policy, int *existed)
{
        if (td < 0) return errNew("td cannot be negative.");
        if (td >= map->cap) return errNew("td is too large.");

        *existed = map->data[td] != NULL;

        if (policy == BB_TD_MAP_OVERWRITE) {
                map->data[td] = v;
                return OK;
        }

        assert(policy == BB_TD_MAP_DO_NOT_OVERWRITE);
        if (!*existed) {
                map->data[td] = v;
        }
        return OK;
}

// -----------------------------------------------------------------------------
// Fn.
// -----------------------------------------------------------------------------

struct bb_fn_t *
bbFnNew()
{
        size_t          psize = sizeof(struct bb_fn_t);
        struct bb_fn_t *p     = malloc(psize);
        memset(p, 0, psize);
        return p;
}

void
bbFnFree(struct bb_fn_t *p)
{
        if (p == NULL) return;
        free(p);
}

// -----------------------------------------------------------------------------
// Passes.
// -----------------------------------------------------------------------------

error_t
runDCEPass(struct bb_fn_t *fn, void *cfg, int *changed)
{
        *changed = 0;
        return OK;
}
