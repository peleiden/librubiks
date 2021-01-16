// Uses hashmap implementation from https://github.com/tidwall/hashmap.c
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "hashmap.h"
#define N 20

struct state {
    char* start;
    long index;
};

typedef struct hashmap hashmap;

uint64_t state_hash(const void* state, uint64_t seed0, uint64_t seed1) {
    const struct state *s = state;
    return hashmap_sip(s->start, N*sizeof(*s->start), seed0, seed1);
}

int state_compare(const void* state1, const void* state2, void* udata) {
    const struct state *s1 = state1;
    const struct state *s2 = state2;
    return memcmp(s1->start, s2->start, N);
}

long unique(long n, char* states, int* unique_index, int* inverse) {
    // TODO: There seems to be a memory error somewhere
    hashmap* map = hashmap_new(sizeof(struct state), 0, 0, 0, state_hash, state_compare, NULL);
    long count = 0;
    for (long i = 0; i < n; i ++) {
        char* p_state = states + N * i;
        struct state s = { .start = p_state, .index = i };
        struct state* state = hashmap_get(map, &s);
        if (state != NULL) {
            inverse[i] = inverse[state->index];
        } else {
            hashmap_set(map, &s);
            unique_index[count] = i;
            inverse[i] = count;
            count ++;
        }
    }
    hashmap_free(map);
    return count;
}

