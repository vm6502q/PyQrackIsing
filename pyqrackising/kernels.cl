#if FP16
#define fwrapper(f, a) (half)f((float)(a))
#define fwrapper2(f, a, b) (half)f((float)(a), (float)(b))
#else
#define fwrapper(f, a) f(a)
#define fwrapper2(f, a, b) f(a, b)
#define INFINITY_R1 INFINITY
#endif
#define INFINITY_R1 ((real1)INFINITY)

__kernel void init_theta(
    __constant real1* fargs,
    const int n_qubits,
    __constant real1* J_eff,
    __constant uint* degrees,
    __global real1* theta
) {
    const int q = get_global_id(0);
    if (q >= n_qubits) {
        return;
    }

    const real1 h_mult = fwrapper(fabs, fargs[2]);
    const real1 J = J_eff[q];
    const uint z = degrees[q];
    const real1 abs_zJ = fwrapper(fabs, z * J);

#if FP16
    theta[q] = (abs_zJ <= EPSILON) ? ((J > ZERO_R1) ? M_PI_F : -M_PI_F) : (real1)asin(fmax(-1.0f, fmin(1.0f, (float)(h_mult / (z * J)))));
#else
    theta[q] = (abs_zJ <= EPSILON) ? ((J > ZERO_R1) ? M_PI_F : -M_PI_F) : asin(fmax(-ONE_R1, fmin(ONE_R1, h_mult / (z * J))));
#endif
}

// By Google Search AI
#if FP16
inline void atomic_add_real1(__global real1* address, real1 val, bool highWord) {
    union {
        uint intVal;
        real1 floatVal[2];
    } oldVal, newVal;

    do {
        oldVal.floatVal[0] = address[0];
        oldVal.floatVal[1] = address[1];
        if (highWord) {
            newVal.floatVal[0] = oldVal.floatVal[0];
            newVal.floatVal[1] = oldVal.floatVal[1] + val;
        } else {
            newVal.floatVal[0] = oldVal.floatVal[0] + val;
            newVal.floatVal[1] = oldVal.floatVal[1];
        }
    } while (atomic_cmpxchg((__global uint*)address, oldVal.intVal, newVal.intVal) != oldVal.intVal);
}
#else
inline void atomic_add_real1(__global real1* address, real1 val) {
    union {
        qint intVal;
        real1 floatVal;
    } oldVal, newVal;

    do {
        oldVal.floatVal = *address;
        newVal.floatVal = oldVal.floatVal + val; // Perform addition, then reinterpret

    } while (atom_cmpxchg((__global qint*)address, oldVal.intVal, newVal.intVal) != oldVal.intVal);
}
#endif

// By Elara (custom OpenAI GPT)
inline real1 probability_by_hamming_weight(
    int q, real1 J, real1 h, uint z, real1 theta, real1 t, int n_qubits
) {
    real1 ratio = fwrapper(fabs, h) / (z * J);
    if (ratio > ONE_R1) {
        ratio = ONE_R1;
    }
    real1 theta_c = fwrapper(asin, ratio);

    real1 p = fwrapper2(pow, TWO_R1, fwrapper(fabs, J / h) - ONE_R1)
        * (ONE_R1 + fwrapper(sin,theta - theta_c) * fwrapper(cos, 1.5f * M_PI_F * J * t + theta) / (ONE_R1 + fwrapper(sqrt, t)))
        - 0.5f;

    real1 numerator = fwrapper2(pow, TWO_R1, (n_qubits + TWO_R1) * p) - ONE_R1;
    real1 denominator = fwrapper2(pow, TWO_R1, p) - ONE_R1;
    real1 result = numerator * fwrapper2(pow, TWO_R1, -((n_qubits + ONE_R1) * p) - p * q) / denominator;

    if (fwrapper(isnan, result) || fwrapper(isinf, result)) {
        return ZERO_R1;
    }

    return result;
}

__kernel void maxcut_hamming_cdf(
    int n_qubits,
    __constant uint* degrees,
    __constant real1* args,
    __constant real1* J_func,
    __constant real1* theta,
    __global real1* hamming_prob
) {
    const real1 delta_t = args[0U];
    const real1 tot_t = args[1U];
    const real1 h_mult = args[2U];
    const int step = get_group_id(0) / n_qubits;
    const int qi = get_group_id(0) % n_qubits;
    const real1 J_eff = J_func[qi];
    const uint z = degrees[qi];
    if (fwrapper(fabs, z * J_eff) <= EPSILON) {
        return;
    }

    const real1 theta_eff = theta[qi];
    const real1 t = step * delta_t;
    const real1 tm1 = (step - 1) * delta_t;
    const real1 h_t = h_mult * (tot_t - t);

    const int n_threads = get_local_size(0);

    for (int qo = get_local_id(0); qo < n_qubits; qo += n_threads) {
        int _qo = (J_eff > ZERO_R1) ? (n_qubits - (1 + qo)) : qo;
        real1 diff = probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, t, n_qubits);
        diff -= probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, tm1, n_qubits);
#if FP16
        atomic_add_real1(&(hamming_prob[_qo & ~1]), diff, _qo & 1U);
#else
        atomic_add_real1(&(hamming_prob[_qo]), diff);
#endif
    }
}

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
    const int i = get_global_id(0);

    // The inputs are chaotic, and this doesn't need to be high-quality, just uniform.
    prng_seed ^= (uint)i;

    real1 energy = INFINITY_R1;

    if (i < combo_count) {
        const int j = i * k;
        // Compute energy for this combination
        energy = bootstrap_worker(best_theta, G_m, indices_array + j, k, n);
    }

    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    // Initialize local memory
    loc_energy[lt_id] = energy;
    loc_index[lt_id] = i;

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
    const int i = get_global_id(0);

    prng_seed ^= (uint)i;

    real1 energy = INFINITY_R1;

    if (i < combo_count) {
        const int j = i * k;
        // Compute energy for this combination
        energy = bootstrap_worker_sparse(best_theta, G_data, G_rows, G_cols, indices_array + j, k, n);
    }

    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    // Initialize local memory
    loc_energy[lt_id] = energy;
    loc_index[lt_id] = i;

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

/// ------ Alternative random sampling rejection kernel by Elara (the OpenAI custom GPT) is below ------ ///

inline real1 rand_uniform(uint *state) {
    return (real1)((double)xorshift32(state) / (double)0xFFFFFFFF);
}

// Compute cut value from bitset solution
real1 compute_cut_bitset(__global const real1* G_m, const uint* sol_bits, int n) {
    real1 cut_val = ZERO_R1;
    for (int u = 0; u < n; u++) {
        int u_word = u >> 5;      // divide by 32
        int u_bit  = u & 31;
        int u_val  = (sol_bits[u_word] >> u_bit) & 1;

        for (int v = u + 1; v < n; v++) {
            int v_word = v >> 5;
            int v_bit  = v & 31;
            int v_val  = (sol_bits[v_word] >> v_bit) & 1;

            if (u_val != v_val) {
                cut_val += G_m[u * n + v];
            }
        }
    }

    return cut_val;
}

#define MIN_WEIGHT ((real1)FLT_EPSILON)
#define MAX_WORDS 4096
#define MAX_WORDS_MASK 4095

__kernel void sample_for_solution_best_bitset(
    __global const real1* G_m,
    __constant real1* thresholds,
    const int n,
    const int shots,
    const real1 max_weight,
    __global real1* rng_seeds,
    __global uint* best_solutions,   // [num_groups × ceil(n/32)]
    __global real1* best_energies,   // [num_groups]
    __local real1* loc_energy,
    __local int* loc_index
) {

    const int gid_orig = get_global_id(0);
    const int lid = get_local_id(0);
    const int group = get_group_id(0);
    const int lsize = get_local_size(0);

    const int words = (n + 31) / 32; // how many uint32s per solution

    uint state = rng_seeds[gid_orig];

    uint sol_bits[MAX_WORDS];
    for (int w = 0; w < words; w++) sol_bits[w] = 0;
    uint temp_sol[MAX_WORDS];

    real1 cut_val = -INFINITY_R1;
    for (int gid = gid_orig; gid < shots; gid += MAX_PROC_ELEM) {
 
        // --- 1. Choose Hamming weight
        real1 mag_prob = rand_uniform(&state);
        int m = 0;
        while (m < n && thresholds[m] < mag_prob) m++;
        m++;

        // --- 2. Build solution bitset
        for (int w = 0; w < words; w++) temp_sol[w] = 0;
        temp_sol[(gid >> 5) & MAX_WORDS_MASK] |= 1U << (gid & 31);

        for (int count = 1; count < m; ++count) {
            real1 highest_weights[TOP_N];
            int best_bits[TOP_N];
            for (int x = 0; x < TOP_N; ++x) {
                highest_weights[x] = -INFINITY_R1;
                best_bits[x] = -1;
            }

            for (int i = 0; i < n; i += 32) {
                for (int j = 0; j < 32; ++j) {
                    const int u = i + j;
                    if (u >= n) {
                        break;
                    }
                    if ((temp_sol[i >> 5] >> j) & 1) {
                        continue;
                    }
                    const int u_offset = u * n;

                    real1 weight = ONE_R1;
                    for (int k = 0; k < n; k += 32) {
                        for (int l = 0; l < 32; ++l) {
                            const int v = k + l;
                            if (v >= n) {
                                break;
                            }
                            if ((temp_sol[k >> 5] >> l) & 1) {
                                weight *= fwrapper2(max, MIN_WEIGHT, ONE_R1 - G_m[u_offset + v] / max_weight);
                            }
                        }
                    }

                    int lowest_option = 0;
                    real1 lowest_weight = highest_weights[0];
                    if (lowest_weight != -INFINITY_R1) {
                        for (int x = 1; x < TOP_N; ++x) {
                            real1 val = highest_weights[x];
                            if (val < lowest_weight) {
                                lowest_option = x;
                                lowest_weight = highest_weights[x];
                                if (val == -INFINITY_R1) {
                                    break;
                                }
                            }
                        }
                    }

                    if (weight > lowest_weight) {
                        highest_weights[lowest_option] = weight;
                        best_bits[lowest_option] = u;
                    }
                }
            }

            real1 total_weight = ZERO_R1;
            for (int x = 0; x < TOP_N; ++x) {
                const real1 val = highest_weights[x];
                if (val == -INFINITY_R1) {
                    continue;
                }
                total_weight += val;
            }

            const real1 bit_prob = rand_uniform(&state);
            real1 tot_prob = ZERO_R1;
            int best_bit = 0;
            for (int x = 0; x < TOP_N; ++x) {
                const real1 val = highest_weights[x];
                if (val == -INFINITY_R1) {
                    continue;
                }
                tot_prob += val;
                if ((total_weight * bit_prob) < tot_prob) {
                    best_bit = best_bits[x];
                    break;
                }
            }

            const int w = best_bit >> 5;
            const int b = best_bit & 31;
            temp_sol[w] |= 1U << b;
        }

        // --- 3. Compute cut value
        real1 temp_cut = compute_cut_bitset(G_m, temp_sol, n);
        if (temp_cut > cut_val) {
            cut_val = temp_cut;
            for (int i = 0; i < words; ++i) {
                sol_bits[i] = temp_sol[i];
            }
        }
    }

    loc_energy[lid] = cut_val;
    loc_index[lid] = gid_orig;
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 4. Reduction for best in workgroup
    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < offset) {
            if (loc_energy[lid + offset] > loc_energy[lid]) {
                loc_energy[lid] = loc_energy[lid + offset];
                loc_index[lid] = loc_index[lid + offset];
            }
        }
    }

    // --- 5. Write best per group
    int winner_gid = loc_index[0];
    if (lid == (winner_gid % get_local_size(0))) {
        best_energies[group] = loc_energy[0];

        // Copy winning bitset solution into best_solutions[group]
        __global uint* dst = best_solutions + group * words;
        for (int w = 0; w < words; w++) {
            dst[w] = sol_bits[w];
        }
    }
}

// Compute cut value from bitset solution
real1 compute_cut_bitset_sparse(__global const real1* G_data, __global const uint* G_rows, __global const uint* G_cols, const uint* sol_bits, int n) {
    real1 cut_val = ZERO_R1;
    for (uint u = 0; u < n; u++) {
        int u_word = u >> 5;      // divide by 32
        int u_bit  = u & 31;
        int u_val  = (sol_bits[u_word] >> u_bit) & 1;

        uint max_col = G_rows[u + 1];
        for (uint col = G_rows[u]; col < max_col; ++col) {
            const int v = G_cols[col];
            int v_word = v >> 5;
            int v_bit  = v & 31;
            int v_val  = (sol_bits[v_word] >> v_bit) & 1;

            if (u_val != v_val) {
                cut_val += G_data[col];
            }
        }
    }

    return cut_val;
}

int binary_search(__global const uint* l, const uint t, const uint len) {
    int left = 0;
    int right = len - 1;

    while (left <= right) {
        int mid = (left + right) >> 1;

        if (l[mid] == t) {
            return mid;
        }

        if (l[mid] < t) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return len;
}

__kernel void sample_for_solution_best_bitset_sparse(
    __global const real1* G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant real1* thresholds,
    const int n,
    const int shots,
    const real1 max_weight,
    __global real1* rng_seeds,
    __global uint* best_solutions,   // [num_groups × ceil(n/32)]
    __global real1* best_energies,   // [num_groups]
    __local real1* loc_energy,
    __local int* loc_index
) {

    const int gid_orig = get_global_id(0);
    const int lid = get_local_id(0);
    const int group = get_group_id(0);
    const int lsize = get_local_size(0);

    const int words = (n + 31) / 32; // how many uint32s per solution

    uint state = rng_seeds[gid_orig];

    uint sol_bits[MAX_WORDS];
    for (int w = 0; w < words; w++) sol_bits[w] = 0;
    uint temp_sol[MAX_WORDS];

    real1 cut_val = -INFINITY_R1;
    for (int gid = gid_orig; gid < shots; gid += MAX_PROC_ELEM) {
 
        // --- 1. Choose Hamming weight
        const real1 mag_prob = rand_uniform(&state);
        int m = 0;
        while (m < n && thresholds[m] < mag_prob) m++;
        m++;

        // --- 2. Build solution bitset
        for (int w = 0; w < words; w++) temp_sol[w] = 0;
        temp_sol[(gid >> 5) & MAX_WORDS_MASK] |= 1U << (gid & 31);

        for (int count = 1; count < m; ++count) {
            real1 highest_weights[TOP_N];
            int best_bits[TOP_N];
            for (int x = 0; x < TOP_N; ++x) {
                highest_weights[x] = -INFINITY_R1;
                best_bits[x] = -1;
            }

            for (int i = 0; i < n; i += 32) {
                for (int j = 0; j < 32; ++j) {
                    const int u = i + j;
                    if (u >= n) {
                        break;
                    }
                    if ((temp_sol[i >> 5] >> j) & 1) {
                        continue;
                    }

                    real1 weight = ONE_R1;

                    uint max_col = G_rows[u + 1];
                    for (int col = G_rows[u]; col < max_col; ++col) {
                        const int v = G_cols[col];

                        if ((temp_sol[v >> 5] >> (v & 31)) & 1) {
                            weight *= fwrapper2(max, MIN_WEIGHT, ONE_R1 - G_data[col] / max_weight);
                        }
                    }

                    for (int v = 0; v < u; ++v) {
                        if (!((temp_sol[v >> 5] >> (v & 31)) & 1)) {
                            continue;
                        }
                        int start = G_rows[v];
                        int end = G_rows[v + 1];
                        int j = binary_search(&(G_cols[start]), u, end - start) + start;
                        if (j < end) {
                            weight *= fwrapper2(max, MIN_WEIGHT, ONE_R1 - G_data[j] / max_weight);
                        }
                    }

                    int lowest_option = 0;
                    real1 lowest_weight = highest_weights[0];
                    if (lowest_weight != -INFINITY_R1) {
                        for (int x = 0; x < TOP_N; ++x) {
                            real1 val = highest_weights[x];
                            if (val < lowest_weight) {
                                lowest_option = x;
                                lowest_weight = highest_weights[x];
                                if (val == -INFINITY_R1) {
                                    break;
                                }
                            }
                        }
                    }

                    if (weight > lowest_weight) {
                        highest_weights[lowest_option] = weight;
                        best_bits[lowest_option] = u;
                    }
                }
            }

            real1 total_weight = ZERO_R1;
            for (int x = 0; x < TOP_N; ++x) {
                const real1 val = highest_weights[x];
                if (val == -INFINITY_R1) {
                    continue;
                }
                total_weight += val;
            }

            const real1 bit_prob = rand_uniform(&state);
            real1 tot_prob = ZERO_R1;
            int best_bit = 0;
            for (int x = 0; x < TOP_N; ++x) {
                const real1 val = highest_weights[x];
                if (val == -INFINITY_R1) {
                    continue;
                }
                tot_prob += val;
                if ((total_weight * bit_prob) < tot_prob) {
                    best_bit = best_bits[x];
                    break;
                }
            }

            const int w = best_bit >> 5;
            const int b = best_bit & 31;
            temp_sol[w] |= 1U << b;
        }

        // --- 3. Compute cut value
        real1 temp_cut = compute_cut_bitset_sparse(G_data, G_rows, G_cols, temp_sol, n);
        if (temp_cut > cut_val) {
            cut_val = temp_cut;
            for (int i = 0; i < words; ++i) {
                sol_bits[i] = temp_sol[i];
            }
        }
    }

    loc_energy[lid] = cut_val;
    loc_index[lid] = gid_orig;
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 4. Reduction for best in workgroup
    for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < offset) {
            if (loc_energy[lid + offset] > loc_energy[lid]) {
                loc_energy[lid] = loc_energy[lid + offset];
                loc_index[lid] = loc_index[lid + offset];
            }
        }
    }

    // --- 5. Write best per group
    int winner_gid = loc_index[0];
    if (lid == (winner_gid % get_local_size(0))) {
        best_energies[group] = loc_energy[0];

        // Copy winning bitset solution into best_solutions[group]
        __global uint* dst = best_solutions + group * words;
        for (int w = 0; w < words; w++) {
            dst[w] = sol_bits[w];
        }
    }
}
