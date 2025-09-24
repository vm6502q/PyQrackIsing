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
    const int i = get_local_id(0);

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
    for (int offset = lt_size / 2; offset > 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lt_id < offset) {
            if (loc_energy[lt_id + offset] < loc_energy[lt_id]) {
                loc_energy[lt_id] = loc_energy[lt_id + offset];
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

float bootstrap_worker_sparse(__constant char* theta, __global double* G_data, __global unsigned* G_rows, __global unsigned* G_cols, __constant unsigned* indices, const int k, const int n) {
    double energy = 0.0;
    for (int u = 0; u < n; ++u) {
        bool u_bit = theta[u];
        for (int x = 0; x < k; ++x) {
            if (indices[x] == u) {
                u_bit = !u_bit;
                break;
            }
        }
        const unsigned mCol = G_rows[u + 1];
        for (unsigned col = G_rows[u]; col < mCol; ++col) {
            const unsigned v = G_cols[col];
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
    const int i = get_local_id(0);

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
    for (int offset = lt_size / 2; offset > 0; offset /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lt_id < offset) {
            if (loc_energy[lt_id + offset] < loc_energy[lt_id]) {
                loc_energy[lt_id] = loc_energy[lt_id + offset];
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
