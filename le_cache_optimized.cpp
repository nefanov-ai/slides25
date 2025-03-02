void le_step_y_cf(le_task *t, const int_t cfs) {
    assert(t->stype == ST_AOS);
    assert(cfs > 0 && cfs <= t->n.x);

    const real k1 = t->dt * t->mat.c1 / t->h.y;
    const real k2 = t->dt * t->mat.c2 / t->h.y;

    le_w *w_2, *w_1, *w, *w1, *w2;
    posix_memalign((void**)&w_2, CACHE_LINE_SIZE, sizeof(le_w) * cfs);
    posix_memalign((void**)&w_1, CACHE_LINE_SIZE, sizeof(le_w) * cfs);
    posix_memalign((void**)&w, CACHE_LINE_SIZE, sizeof(le_w) * cfs);
    posix_memalign((void**)&w1, CACHE_LINE_SIZE, sizeof(le_w) * cfs);
    posix_memalign((void**)&w2, CACHE_LINE_SIZE, sizeof(le_w) * cfs);

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < t->n.x; k += cfs) {
        int cfs_n = (k + cfs > t->n.x) ? t->n.x - k : cfs;

        // Initialize buffers
        for (int i = 0; i < cfs_n; i++) {
            omega_y(&t->mat, &gind(i + k, 0), &w[i]);
            omega_y(&t->mat, &gind(i + k, 1), &w1[i]);
            omega_y(&t->mat, &gind(i + k, 2), &w2[i]);
        }
        memcpy(w_2, w, sizeof(le_w) * cfs_n);
        memcpy(w_1, w, sizeof(le_w) * cfs_n);

        // Process grid in chunks
        for (int j = 0; j < t->n.y; j++) {
            for (int i = 0; i < cfs_n; i++) {
                le_w d;
                reconstruct(w_2[i], w_1[i], w[i], w1[i], w2[i], k1, k2, &d);
                inc_y(&t->mat, &gind(i + k, j), &d);

                // Prefetch data for the next iteration
                if (j < t->n.y - 3) {
                    __builtin_prefetch(&gind(i + k, j + 3), 0, 1);
                }
            }

            // Swap buffers
            le_w *wt = w_2;
            w_2 = w_1;
            w_1 = w;
            w = w1;
            w1 = w2;
            w2 = wt;

            // Load new data for the next chunk
            if (j < t->n.y - 3) {
                for (int i = 0; i < cfs_n; i++) {
                    omega_y(&t->mat, &gind(i + k, j + 3), &w2[i]);
                }
            }
        }
    }

    free(w_2);
    free(w_1);
    free(w);
    free(w1);
    free(w2);
}
