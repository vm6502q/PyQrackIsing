__kernel void init_theta(
    __constant float* fargs,
    const int n_qubits,
    __constant float* J_eff,
    __constant float* degrees,
    __global float* theta
) {
    const int q = get_global_id(0);
    if (q >= n_qubits) {
        return;
    }

    const float h_mult = fabs(fargs[2]);

    const float J = J_eff[q];
    const float z = degrees[q];
    const float abs_zJ = fabs(z * J);

    float val;
    if (abs_zJ < 2e-40f) {
        val = (J > 0.0f) ? M_PI_F : -M_PI_F;
    } else {
        val = asin(fmax(-1.0f, fmin(1.0f, h_mult / (z * J))));
    }

    theta[q] = val;
}

// By Google Search AI
inline void AtomicAdd_g_f(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } oldVal, newVal;

    do {
        oldVal.floatVal = *source;
        newVal.floatVal = oldVal.floatVal + operand;
        // Use atomic_cmpxchg to atomically update the value
    } while (atomic_cmpxchg((volatile __global unsigned int*)source, oldVal.intVal, newVal.intVal) != oldVal.intVal);
}

// By Elara (custom OpenAI GPT)
inline float probability_by_hamming_weight(
    int q, float J, float h, float z, float theta, float t, int n_qubits
) {
    float ratio = fabs(h) / (z * J);
    if (ratio > 1.0f) ratio = 1.0f;
    float theta_c = asin(ratio);

    float p = pow(2.0f, fabs(J / h) - 1.0f)
        * (1.0f + sin(theta - theta_c) * cos(1.5f * M_PI_F * J * t + theta) / (1.0f + sqrt(t)))
        - 0.5f;

    if ((p * (n_qubits + 2.0f)) >= 1024.0f) return 0.0f;

    float numerator = pow(2.0f, (n_qubits + 2.0f) * p) - 1.0f;
    float denominator = pow(2.0f, p) - 1.0f;
    float result = numerator * pow(2.0f, -((n_qubits + 1.0f) * p) - p * q) / denominator;

    if (isnan(result) || isinf(result)) return 0.0f;
    return result;
}

__kernel void maxcut_hamming_cdf(
    int n_qubits,
    __constant int* degrees,
    __constant float* args,
    __constant float* J_func,
    __constant float* theta,
    __global float* hamming_prob
) {
    const float delta_t = args[0U];
    const float tot_t = args[1U];
    const float h_mult = args[2U];
    const int step = get_group_id(0) / n_qubits;
    const int qi = get_group_id(0) % n_qubits;
    const float J_eff = J_func[qi];
    const float z = degrees[qi];
    if (fabs(z * J_eff) <= (pow(2.0f, -52))) return;

    const float theta_eff = theta[qi];
    const float t = step * delta_t;
    const float tm1 = (step - 1) * delta_t;
    const float h_t = h_mult * (tot_t - t);

    const int n_threads = get_local_size(0);

    for (int qo = get_local_id(0); qo < n_qubits; qo += n_threads) {
        int _qo = (J_eff > 0.0f) ? (n_qubits - (1 + qo)) : qo;
        float diff = probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, t, n_qubits);
        diff -= probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, tm1, n_qubits);
        AtomicAdd_g_f(&(hamming_prob[_qo]), diff);
    }
}

float bootstrap_worker(__constant char* theta, __global double* G_m, __constant int* indices, const int k, const int n) {
    double energy = 0.0;
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
            const double val = G_m[u_offset + v];
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

    return (float)energy;
}

__kernel void bootstrap(
    __global double* G_m,
    __constant char* best_theta,
    __constant int* indices_array,
    __constant int* args,               // args[0] = n, args[1] = k
    __global float* min_energy_ptr,     // output: per-group min energy
    __global int* min_index_ptr,        // output: per-group best index (i)
    __local float* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const int k = args[1];
    const int combo_count = args[2];
    const int i = get_global_id(0);

    float energy = INFINITY;

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
        if ((lt_id < offset) && (loc_energy[lt_id + offset] < loc_energy[lt_id])) {
            loc_energy[lt_id] = loc_energy[lt_id + offset];
            loc_index[lt_id] = loc_index[lt_id + offset];
        }
    }

    // Write out per-group result
    if (lt_id == 0) {
        min_energy_ptr[get_group_id(0)] = loc_energy[0];
        min_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

float bootstrap_worker_sparse(__constant char* theta, __global double* G_data, __global unsigned* G_rows, __global unsigned* G_cols, __constant int* indices, const int k, const int n) {
    double energy = 0.0;
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
            const double val = G_data[col];
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

    return (float)energy;
}

__kernel void bootstrap_sparse(
    __global double* G_data,
    __global unsigned* G_rows,
    __global unsigned* G_cols,
    __constant char* best_theta,
    __constant int* indices_array,
    __constant int* args,               // args[0] = n, args[1] = k
    __global float* min_energy_ptr,     // output: per-group min energy
    __global int* min_index_ptr,        // output: per-group best index (i)
    __local float* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const int k = args[1];
    const int combo_count = args[2];
    const int i = get_global_id(0);

    float energy = INFINITY;

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
        if ((lt_id < offset) && (loc_energy[lt_id + offset] < loc_energy[lt_id])) {
            loc_energy[lt_id] = loc_energy[lt_id + offset];
            loc_index[lt_id] = loc_index[lt_id + offset];
        }
    }

    // Write out per-group result
    if (lt_id == 0) {
        min_energy_ptr[get_group_id(0)] = loc_energy[0];
        min_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

/// ------ Alternative random sampling rejection kernel by Elara (the OpenAI custom GPT) is below ------ ///

inline uint xorshift32(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

inline float rand_uniform(uint *state) {
    return (float)xorshift32(state) / (float)0xFFFFFFFF;
}

// Compute cut value from bitset solution
double compute_cut_bitset(__global const double* G_m, const uint* sol_bits, int n) {
    double cut_val = 0.0;
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

#define MAX_WORDS 4096
#define MAX_WORDS_MASK 4095

__kernel void sample_for_solution_best_bitset(
    __global const double* G_m,
    __global const float* thresholds,
    const int n,
    const int shots,
    const double max_weight,
    __global float* rng_seeds,
    __global uint* best_solutions,   // [num_groups × ceil(n/32)]
    __global float* best_energies,   // [num_groups]
    __local float* loc_energy,
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

    double cut_val = -INFINITY;
    for (int gid = gid_orig; gid < shots; gid += MAX_PROC_ELEM) {
 
        // --- 1. Choose Hamming weight
        float mag_prob = rand_uniform(&state);
        int m = 0;
        while (m < n && thresholds[m] < mag_prob) m++;
        m++;

        // --- 2. Build solution bitset
        for (int w = 0; w < words; w++) temp_sol[w] = 0;
        temp_sol[(gid >> 5) & MAX_WORDS_MASK] |= 1U << (gid & 31);

        for (int count = 1; count < m; ++count) {
            double highest_weights[TOP_N];
            int best_bits[TOP_N];
            for (int x = 0; x < TOP_N; ++x) {
                highest_weights[x] = -INFINITY;
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

                    double weight = 1.0;
                    for (int k = 0; k < n; k += 32) {
                        for (int l = 0; l < 32; ++l) {
                            const int v = k + l;
                            if (v >= n) {
                                break;
                            }
                            if ((temp_sol[k >> 5] >> l) & 1) {
                                weight *= max(2e-7, 1.0 - G_m[u_offset + v] / max_weight);
                            }
                        }
                    }

                    int lowest_option = 0;
                    double lowest_weight = highest_weights[0];
                    if (lowest_weight != -INFINITY) {
                        for (int x = 0; x < TOP_N; ++x) {
                            double val = highest_weights[x];
                            if (val < lowest_weight) {
                                lowest_option = x;
                                lowest_weight = highest_weights[x];
                                if (val == -INFINITY) {
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

            double total_weight = 0.0;
            for (int x = 0; x < TOP_N; ++x) {
                const double val = highest_weights[x];
                if (val == -INFINITY) {
                    continue;
                }
                total_weight += val;
            }

            const float bit_prob = rand_uniform(&state);
            double tot_prob = 0.0;
            int best_bit = 0;
            for (int x = 0; x < TOP_N; ++x) {
                const double val = highest_weights[x];
                if (val == -INFINITY) {
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
        double temp_cut = compute_cut_bitset(G_m, temp_sol, n);
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
double compute_cut_bitset_sparse(__global const double* G_data, __global const unsigned* G_rows, __global const unsigned* G_cols, const uint* sol_bits, int n) {
    double cut_val = 0.0;
    for (unsigned u = 0; u < n; u++) {
        int u_word = u >> 5;      // divide by 32
        int u_bit  = u & 31;
        int u_val  = (sol_bits[u_word] >> u_bit) & 1;

        unsigned max_col = G_rows[u + 1];
        for (unsigned col = G_rows[u]; col < max_col; ++col) {
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

int binary_search(__global const unsigned* l, const unsigned t, const unsigned len) {
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
    __global const double* G_data,
    __global const unsigned* G_rows,
    __global const unsigned* G_cols,
    __global const float* thresholds,
    const int n,
    const int shots,
    const double max_weight,
    __global float* rng_seeds,
    __global uint* best_solutions,   // [num_groups × ceil(n/32)]
    __global float* best_energies,   // [num_groups]
    __local float* loc_energy,
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

    double cut_val = -INFINITY;
    for (int gid = gid_orig; gid < shots; gid += MAX_PROC_ELEM) {
 
        // --- 1. Choose Hamming weight
        const float mag_prob = rand_uniform(&state);
        int m = 0;
        while (m < n && thresholds[m] < mag_prob) m++;
        m++;

        // --- 2. Build solution bitset
        for (int w = 0; w < words; w++) temp_sol[w] = 0;
        temp_sol[(gid >> 5) & MAX_WORDS_MASK] |= 1U << (gid & 31);

        for (int count = 1; count < m; ++count) {
            double highest_weights[TOP_N];
            int best_bits[TOP_N];
            for (int x = 0; x < TOP_N; ++x) {
                highest_weights[x] = -INFINITY;
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

                    double weight = 1.0;

                    unsigned max_col = G_rows[u + 1];
                    for (int col = G_rows[u]; col < max_col; ++col) {
                        const int v = G_cols[col];

                        if ((temp_sol[v >> 5] >> (v & 31)) & 1) {
                            weight *= max(2e-7, 1.0 - G_data[col] / max_weight);
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
                            weight *= max(2e-7, 1.0 - G_data[j] / max_weight);
                        }
                    }

                    int lowest_option = 0;
                    double lowest_weight = highest_weights[0];
                    if (lowest_weight != -INFINITY) {
                        for (int x = 0; x < TOP_N; ++x) {
                            double val = highest_weights[x];
                            if (val < lowest_weight) {
                                lowest_option = x;
                                lowest_weight = highest_weights[x];
                                if (val == -INFINITY) {
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

            double total_weight = 0.0;
            for (int x = 0; x < TOP_N; ++x) {
                const double val = highest_weights[x];
                if (val == -INFINITY) {
                    continue;
                }
                total_weight += val;
            }

            const float bit_prob = rand_uniform(&state);
            double tot_prob = 0.0;
            int best_bit = 0;
            for (int x = 0; x < TOP_N; ++x) {
                const double val = highest_weights[x];
                if (val == -INFINITY) {
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
        double temp_cut = compute_cut_bitset_sparse(G_data, G_rows, G_cols, temp_sol, n);
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
