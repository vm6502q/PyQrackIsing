#if FP16
#define fwrapper(f, a) (half)f((float)(a))
#define fwrapper2(f, a, b) (half)f((float)(a), (float)(b))
#else
#define fwrapper(f, a) f(a)
#define fwrapper2(f, a, b) f(a, b)
#endif

inline uint xorshift32(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

real1 bootstrap_worker(__constant char* theta, __global const real1* G_m, __constant int* indices, const int k, const int n) {
    real1 energy = ZERO_R1;
    const size_t n_st = (size_t)n;
    for (int u = 0; u < n; ++u) {
        const size_t u_offset = u * n_st;
        bool u_bit = theta[u];
        for (int x = 0; x < k; ++x) {
            if (indices[x] == u) {
                u_bit = !u_bit;
                break;
            }
        }
        for (int v = u + 1; v < n; ++v) {
            const real1 val = G_m[u_offset + v];
            bool v_bit = theta[v];
            for (int x = 0; x < k; ++x) {
                if (indices[x] == v) {
                    v_bit = !v_bit;
                    break;
                }
            }
            energy += (u_bit == v_bit) ? val : -val;
        }
    }

    return energy;
}

__kernel void bootstrap(
    uint prng_seed,
    __global const real1* G_m,
    __constant char* best_theta,
    __constant int* indices_array,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* min_energy_ptr,     // output: per-group min energy
    __global int* min_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const int k = args[1];
    const int combo_count = args[2];
    int i = get_global_id(0);

    // The inputs are chaotic, and this doesn't need to be high-quality, just uniform.
    prng_seed ^= (uint)i;

    real1 best_energy = INFINITY;
    int best_i = i;

    for (; i < combo_count; i += MAX_PROC_ELEM) {
        const int j = i * k;
        const real1 energy = bootstrap_worker(best_theta, G_m, indices_array + j, k, n);
        if (energy < best_energy) {
            best_energy = energy;
            best_i = i;
        } else if (((energy - best_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
            best_i = i;
        }
    }

    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    loc_energy[lt_id] = best_energy;
    loc_index[lt_id] = best_i;

    // Reduce within workgroup
    for (int offset = lt_size >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lt_id < offset) {
            real1 hid_energy = loc_energy[lt_id + offset];
            real1 lid_energy = loc_energy[lt_id];
            if (hid_energy < lid_energy) {
                loc_energy[lt_id] = hid_energy;
                loc_index[lt_id] = loc_index[lt_id + offset];
            } else if (((hid_energy - lid_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
                loc_index[lt_id] = loc_index[lt_id + offset];
            }
        }
    }

    // Write out per-group result
    if (lt_id == 0) {
        min_energy_ptr[get_group_id(0)] = loc_energy[0];
        min_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

real1 bootstrap_worker_sparse(__constant char* theta, __global const real1* G_data, __global const uint* G_rows, __global const uint* G_cols, __constant int* indices, const int k, const int n) {
    real1 energy = ZERO_R1;
    for (int u = 0; u < n; ++u) {
        bool u_bit = theta[u];
        for (int x = 0; x < k; ++x) {
            if (indices[x] == u) {
                u_bit = !u_bit;
                break;
            }
        }
        const size_t mCol = G_rows[u + 1];
        for (int col = G_rows[u]; col < mCol; ++col) {
            const int v = G_cols[col];
            const real1 val = G_data[col];
            bool v_bit = theta[v];
            for (int x = 0; x < k; ++x) {
                if (indices[x] == v) {
                    v_bit = !v_bit;
                    break;
                }
            }
            energy += (u_bit == v_bit) ? val : -val;
        }
    }

    return energy;
}

__kernel void bootstrap_sparse(
    uint prng_seed,
    __global const real1* G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant char* best_theta,
    __constant int* indices_array,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* min_energy_ptr,     // output: per-group min energy
    __global int* min_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const int k = args[1];
    const int combo_count = args[2];
    int i = get_global_id(0);

    prng_seed ^= (uint)i;

    real1 best_energy = INFINITY;
    int best_i = i;

    for (; i < combo_count; i += MAX_PROC_ELEM) {
        const int j = i * k;
        const real1 energy = bootstrap_worker_sparse(best_theta, G_data, G_rows, G_cols, indices_array + j, k, n);
        if (energy < best_energy) {
            best_energy = energy;
            best_i = i;
        } else if (((energy - best_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
            best_i = i;
        }
    }

    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    loc_energy[lt_id] = best_energy;
    loc_index[lt_id] = best_i;

    // Reduce within workgroup
    for (int offset = lt_size >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lt_id < offset) {
            real1 hid_energy = loc_energy[lt_id + offset];
            real1 lid_energy = loc_energy[lt_id];
            if (hid_energy < lid_energy) {
                loc_energy[lt_id] = hid_energy;
                loc_index[lt_id] = loc_index[lt_id + offset];
            } else if (((hid_energy - lid_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
                loc_index[lt_id] = loc_index[lt_id + offset];
            }
        }
    }

    // Write out per-group result
    if (lt_id == 0) {
        min_energy_ptr[get_group_id(0)] = loc_energy[0];
        min_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

// Helper to read from segmented G_m
inline real1 get_G_m(
    __global const real1** G_m,
    size_t flat_idx,
    int segment_size
) {
    return G_m[flat_idx / segment_size][flat_idx % segment_size];
}

real1 bootstrap_worker_segmented(
    __constant char* theta,
    __global const real1** G_m,
    __constant int* indices,
    const int k,
    const int n,
    const int segment_size
) {
    real1 energy = ZERO_R1;
    const size_t n_st = (size_t)n;

    for (int u = 0; u < n; ++u) {
        const size_t u_offset = u * n_st;
        bool u_bit = theta[u];
        for (int x = 0; x < k; ++x) {
            if (indices[x] == u) {
                u_bit = !u_bit;
                break;
            }
        }
        for (int v = u + 1; v < n; ++v) {
            size_t flat_idx = u_offset + v;
            real1 val = get_G_m(G_m, flat_idx, segment_size);

            bool v_bit = theta[v];
            for (int x = 0; x < k; ++x) {
                if (indices[x] == v) {
                    v_bit = !v_bit;
                    break;
                }
            }
            energy += (u_bit == v_bit) ? val : -val;
        }
    }

    return energy;
}


__kernel void bootstrap_segmented(
    uint prng_seed,
    __global const real1* G_m0,
    __global const real1* G_m1,
    __global const real1* G_m2,
    __global const real1* G_m3,
    __constant char* best_theta,
    __constant int* indices_array,
    __constant int* args,               // args[0]=n, args[1]=k, args[2]=combo_count, args[3]=segment_size
    __global real1* min_energy_ptr,
    __global int* min_index_ptr,
    __local real1* loc_energy,
    __local int* loc_index
) {
    __global const real1* G_m[4] = { G_m0, G_m1, G_m2, G_m3 };

    const int n = args[0];
    const int k = args[1];
    const int combo_count = args[2];
    const int segment_size = args[3];
    int i = get_global_id(0);

    prng_seed ^= (uint)i;

    real1 best_energy = INFINITY;
    int best_i = i;

    for (; i < combo_count; i += MAX_PROC_ELEM) {
        const int j = i * k;
        const real1 energy = bootstrap_worker_segmented(best_theta, G_m, indices_array + j, k, n, segment_size);
        if (energy < best_energy) {
            best_energy = energy;
            best_i = i;
        } else if (((energy - best_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
            best_i = i;
        }
    }

    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    loc_energy[lt_id] = best_energy;
    loc_index[lt_id] = best_i;

    for (int offset = lt_size >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lt_id < offset) {
            real1 hid_energy = loc_energy[lt_id + offset];
            real1 lid_energy = loc_energy[lt_id];
            if (hid_energy < lid_energy) {
                loc_energy[lt_id] = hid_energy;
                loc_index[lt_id] = loc_index[lt_id + offset];
            } else if (((hid_energy - lid_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
                loc_index[lt_id] = loc_index[lt_id + offset];
            }
        }
    }

    if (lt_id == 0) {
        min_energy_ptr[get_group_id(0)] = loc_energy[0];
        min_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

real1 bootstrap_worker_sparse_segmented(
    __constant char* theta,
    __global const real1** G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant int* indices,
    const int k,
    const int n,
    const int segment_size
) {
    real1 energy = ZERO_R1;

    for (int u = 0; u < n; ++u) {
        bool u_bit = theta[u];
        for (int x = 0; x < k; ++x) {
            if (indices[x] == u) {
                u_bit = !u_bit;
                break;
            }
        }

        const uint row_start = G_rows[u];
        const uint row_end = G_rows[u + 1];

        for (uint col = row_start; col < row_end; ++col) {
            const int v = G_cols[col];
            real1 val = get_G_m(G_data, col, segment_size);

            bool v_bit = theta[v];
            for (int x = 0; x < k; ++x) {
                if (indices[x] == v) {
                    v_bit = !v_bit;
                    break;
                }
            }

            energy += (u_bit == v_bit) ? val : -val;
        }
    }

    return energy;
}

__kernel void bootstrap_sparse_segmented(
    uint prng_seed,
    __global const real1* G_data0,
    __global const real1* G_data1,
    __global const real1* G_data2,
    __global const real1* G_data3,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant char* best_theta,
    __constant int* indices_array,
    __constant int* args,               // args[0] = n, args[1] = k, args[2] = combo_count, args[3] = segment_size
    __global real1* min_energy_ptr,
    __global int* min_index_ptr,
    __local real1* loc_energy,
    __local int* loc_index
) {
    __global const real1* G_data[4] = { G_data0, G_data1, G_data2, G_data3 };

    const int n = args[0];
    const int k = args[1];
    const int combo_count = args[2];
    const int segment_size = args[3];
    int i = get_global_id(0);

    prng_seed ^= (uint)i;

    real1 best_energy = INFINITY;
    int best_i = i;

    for (; i < combo_count; i += MAX_PROC_ELEM) {
        const int j = i * k;
        const real1 energy = bootstrap_worker_sparse_segmented(best_theta, G_data, G_rows, G_cols, indices_array + j, k, n, segment_size);
        if (energy < best_energy) {
            best_energy = energy;
            best_i = i;
        } else if (((energy - best_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
            best_i = i;
        }
    }

    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    loc_energy[lt_id] = best_energy;
    loc_index[lt_id] = best_i;

    for (int offset = lt_size >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lt_id < offset) {
            real1 hid_energy = loc_energy[lt_id + offset];
            real1 lid_energy = loc_energy[lt_id];
            if (hid_energy < lid_energy) {
                loc_energy[lt_id] = hid_energy;
                loc_index[lt_id] = loc_index[lt_id + offset];
            } else if (((hid_energy - lid_energy) <= EPSILON) && ((xorshift32(&prng_seed) >> 31) & 1)) {
                loc_index[lt_id] = loc_index[lt_id + offset];
            }
        }
    }

    if (lt_id == 0) {
        min_energy_ptr[get_group_id(0)] = loc_energy[0];
        min_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

