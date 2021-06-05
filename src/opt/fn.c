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

#define dictSetKey(d, entry, _key_)                                  \
        do {                                                         \
                if ((d)->type->keyDup)                               \
                        (entry)->key =                               \
                            (d)->type->keyDup((d)->privdata, _key_); \
                else                                                 \
                        (entry)->key = (_key_);                      \
        } while (0)

#define dictSetVal(d, entry, _val_)                                  \
        do {                                                         \
                if ((d)->type->valDup)                               \
                        (entry)->v.val =                             \
                            (d)->type->valDup((d)->privdata, _val_); \
                else                                                 \
                        (entry)->v.val = (_val_);                    \
        } while (0)

/* Expand the hash table if needed */
static error_t
_dictExpandIfNeeded(struct dict_t *d)
{
        /* If the hash table is empty expand it to the initial size. */
        if (d->ht.size == 0) return dictExpand(d, DICT_HT_INITIAL_SIZE);

        // TODO
        //
        // /* If we reached the 1:1 ratio, and we are allowed to resize the hash
        //  * table (global setting) or we should avoid it but the ratio between
        //  * elements/buckets is over the "safe" threshold, we resize doubling
        //  * the number of buckets. */
        // if (d->ht[0].used >= d->ht[0].size &&
        //     (dict_can_resize ||
        //      d->ht[0].used/d->ht[0].size > dict_force_resize_ratio) &&
        //     dictTypeExpandAllowed(d))
        // {
        //     return dictExpand(d, d->ht[0].used + 1);
        // }
        return OK;
}

#define dictCompareKeys(d, key1, key2)                                      \
        (((d)->type->keyCmp) ? (d)->type->keyCmp((d)->privdata, key1, key2) \
                             : (key1) == (key2))

#define dictHashKey(d, key) (d)->type->hashFn(key)

/* Returns the index of a free slot that can be populated with
 * a hash entry for the given 'key'.
 * If the key already exists, -1 is returned
 * and the optional output parameter may be filled.
 */
static long
_dictKeyIndex(struct dict_t *d, const void *key, uint64_t hash,
              struct dict_entry_t **existing)
{
        unsigned long        idx;
        struct dict_entry_t *he;
        if (existing) *existing = NULL;

        /* Expand the hash table if needed */
        if (_dictExpandIfNeeded(d))
                return errEmitNote("failed to expand the table.");

        idx = hash & d->ht.sizemask;
        /* Search if this slot does not already contain the given key */
        he = d->ht.table[idx];
        while (he != NULL) {
                if (key == he->key || dictCompareKeys(d, key, he->key)) {
                        if (existing) *existing = he;
                        return -1;
                }
                he = he->next;
        }
        return idx;
}

/*
 * Low level add or find:
 *
 * This function adds the entry but instead of setting a value returns the
 * dictEntry structure to the user, that will make sure to fill the value
 * field as he wishes.
 *
 * This function is also directly exposed to the user API to be called
 * mainly in order to store non-pointers inside the hash value, example:
 *
 * entry = dictAddRaw(dict,mykey,NULL);
 * if (entry != NULL) dictSetSignedIntegerVal(entry,1000);
 *
 * Return values:
 *
 * If key already exists NULL is returned, and "*existing" is populated
 * with the existing entry if existing is not NULL.
 *
 * If key was added, the hash entry is returned to be manipulated by the caller.
 */
struct dict_entry_t *
dictAddRaw(struct dict_t *d, void *key, struct dict_entry_t **existing)
{
        long                 index;
        struct dict_entry_t *entry;
        struct dict_table_t *ht = &d->ht;

        /* Get the index of the new element, or -1 if the element already
         * exists. */
        if ((index = _dictKeyIndex(d, key, dictHashKey(d, key), existing)) ==
            -1)
                return NULL;

        /* Allocate the memory and store the new entry.
         * Insert the element in top, with the assumption that in a database
         * system it is more likely that recently added entries are accessed
         * more frequently. */
        entry            = malloc(sizeof(*entry));
        entry->next      = ht->table[index];
        ht->table[index] = entry;

        /* Set the hash entry fields. */
        dictSetKey(d, entry, key);
        return entry;
}

error_t
dictAdd(struct dict_t *d, void *key, void *val)
{
        struct dict_entry_t *entry = dictAddRaw(d, key, NULL);

        if (!entry) return errNew("entry already existed.");
        dictSetVal(d, entry, val);
        return OK;
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
