void reduce_energy_index(real1 energy, int i, __local real1* loc_energy, __local int* loc_index, __global real1* max_energy_ptr, __global int* max_index_ptr) {
    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    loc_energy[lt_id] = energy;
    loc_index[lt_id] = i;

    // Reduce within workgroup
    for (int offset = lt_size >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        real1 hid_energy, lid_energy;
        if (lt_id < offset) {
            hid_energy = loc_energy[lt_id + offset];
            lid_energy = loc_energy[lt_id];
            if (hid_energy > lid_energy) {
                loc_energy[lt_id] = hid_energy;
                loc_index[lt_id] = loc_index[lt_id + offset];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write out per-group result
    if (lt_id == 0) {
        max_energy_ptr[get_group_id(0)] = loc_energy[0];
        max_index_ptr[get_group_id(0)] = loc_index[0];
    }
}

inline bool get_bit(__global const uint* theta, const size_t u) {
    return (theta[u >> 5U] >> (u & 31U)) & 1U;
}

// Helper to read from segmented G_m
inline real1 get_G_m(
    __global const real1** G_m,
    size_t flat_idx,
    int segment_size
) {
    return G_m[flat_idx / segment_size][flat_idx % segment_size];
}

real1 cut_worker(__global const uint* theta, __global const real1* G_m, const int n, const bool is_spin_glass) {
    real1 energy = ZERO_R1;
    const size_t n_st = (size_t)n;
    for (int u = 0; u < n; ++u) {
        const size_t u_offset = u * n_st;
        const bool u_bit = get_bit(theta, u);
        for (int v = u + 1; v < n; ++v) {
            const real1 val = G_m[u_offset + v];
            const bool v_bit = get_bit(theta, v);
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void calculate_cut(
    __global const real1* G_m,
    __global uint* theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const int shots = args[1];
    const bool is_spin_glass = args[2];
    const int n32 = (n + 31) >> 5U;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < shots; i += max_i) {
        const int j = i * n32;
        const real1 energy = cut_worker(theta + j, G_m, n, is_spin_glass);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

inline bool get_const_bit(__constant uint* theta, const size_t u) {
    return (theta[u >> 5U] >> (u & 31U)) & 1U;
}

real1 single_bit_flip_worker(__constant uint* theta, __global const real1* G_m, const int n, const bool is_spin_glass, const int k) {
    const size_t k_offset = k * (size_t)n;
    const bool k_bit = !get_const_bit(theta, k);
    real1 energy = ZERO_R1;
    for (int v = 0; v < k; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = G_m[k_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
    }
    for (int v = k + 1; v < n; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = G_m[k_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
    }

    return energy;
}

__kernel void single_bit_flips(
    __global const real1* G_m,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const bool is_spin_glass = args[1];

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < n; i += max_i) {
        const real1 energy = single_bit_flip_worker(best_theta, G_m, n, is_spin_glass, i);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 double_bit_flip_worker(__constant uint* theta, __global const real1* G_m, const int n, const bool is_spin_glass, int k, int l) {
    if (l < k) {
        int t = k;
        k = l;
        l = t;
    }
    const size_t k_offset = k * (size_t)n;
    const bool k_bit = !get_const_bit(theta, k);
    const size_t l_offset = l * (size_t)n;
    const bool l_bit = !get_const_bit(theta, l);
    real1 energy = ZERO_R1;
    for (int v = 0; v < k; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = G_m[k_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
        val = G_m[l_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (l_bit != v_bit) ? val : -val;
    }
    for (int v = k + 1; v < l; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = G_m[k_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
        val = G_m[l_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (l_bit != v_bit) ? val : -val;
    }
    for (int v = l + 1; v < n; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = G_m[k_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
        val = G_m[l_offset + v];
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (l_bit != v_bit) ? val : -val;
    }

    return energy;
}

__kernel void double_bit_flips(
    __global const real1* G_m,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int combo_count = (n * (n - 1)) >> 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < combo_count; i += max_i) {
        int c = i;
        int k = 0;
        int lcv = n - 1;
        while (c >= lcv) {
            c -= lcv;
            ++k;
            --lcv;

            if (!lcv) {
                break;
            }
        }
        const int l = c + k + 1;

        const real1 energy = double_bit_flip_worker(best_theta, G_m, n, is_spin_glass, k, l);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 cut_worker_sparse(__global const uint* theta, __global const real1* G_data, __global const uint* G_rows, __global const uint* G_cols, const int n, const bool is_spin_glass) {
    real1 energy = ZERO_R1;
    for (int u = 0; u < n; ++u) {
        const bool u_bit = get_bit(theta, u);
        const size_t mCol = G_rows[u + 1];
        for (int col = G_rows[u]; col < mCol; ++col) {
            const int v = G_cols[col];
            const real1 val = G_data[col];
            const bool v_bit = get_bit(theta, v);
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void calculate_cut_sparse(
    __global const real1* G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __global const uint* theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const int shots = args[1];
    const bool is_spin_glass = args[2];
    const int n32 = (n + 31) >> 5U;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < shots; i += max_i) {
        const int j = i * n32;
        const real1 energy = cut_worker_sparse(theta + j, G_data, G_rows, G_cols, n, is_spin_glass);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 single_bit_flip_worker_sparse(__constant uint* theta, __global const real1* G_data, __global const uint* G_rows, __global const uint* G_cols, const int n, const bool is_spin_glass, const int k) {
    real1 energy = ZERO_R1;
    for (int u = 0; u < n; ++u) {
        bool u_bit = get_const_bit(theta, u);
        if (u == k) {
            u_bit = !u_bit;
        }
        const size_t mCol = G_rows[u + 1];
        for (int col = G_rows[u]; col < mCol; ++col) {
            const int v = G_cols[col];
            const real1 val = G_data[col];
            bool v_bit = get_const_bit(theta, v);
            if (v == k) {
                v_bit = !v_bit;
            }
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void single_bit_flips_sparse(
    __global const real1* G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const bool is_spin_glass = args[1];

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < n; i += max_i) {
        const real1 energy = single_bit_flip_worker_sparse(best_theta, G_data, G_rows, G_cols, n, is_spin_glass, i);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 double_bit_flip_worker_sparse(__constant uint* theta, __global const real1* G_data, __global const uint* G_rows, __global const uint* G_cols, const int n, const bool is_spin_glass, const int k, const int l) {
    real1 energy = ZERO_R1;
    for (int u = 0; u < n; ++u) {
        bool u_bit = get_const_bit(theta, u);
        if ((u == k) || (u == l)) {
            u_bit = !u_bit;
        }
        const size_t mCol = G_rows[u + 1];
        for (int col = G_rows[u]; col < mCol; ++col) {
            const int v = G_cols[col];
            const real1 val = G_data[col];
            bool v_bit = get_const_bit(theta, v);
            if ((v == k) || (v == l)) {
                v_bit = !v_bit;
            }
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void double_bit_flips_sparse(
    __global const real1* G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int combo_count = (n * (n - 1)) >> 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < combo_count; i += max_i) {
        int c = i;
        int k = 0;
        int lcv = n - 1;
        while (c >= lcv) {
            c -= lcv;
            ++k;
            --lcv;

            if (!lcv) {
                break;
            }
        }
        const int l = c + k + 1;

        const real1 energy = double_bit_flip_worker_sparse(best_theta, G_data, G_rows, G_cols, n, is_spin_glass, k, l);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 cut_worker_segmented(
    __global const uint* theta,
    __global const real1** G_m,
    const int n,
    const int segment_size,
    const bool is_spin_glass
) {
    real1 energy = ZERO_R1;
    const size_t n_st = (size_t)n;

    for (int u = 0; u < n; ++u) {
        const size_t u_offset = u * n_st;
        const bool u_bit = get_bit(theta, u);
        for (int v = u + 1; v < n; ++v) {
            const size_t flat_idx = u_offset + v;
            const real1 val = get_G_m(G_m, flat_idx, segment_size);
            const bool v_bit = get_bit(theta, v);
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}


__kernel void calculate_cut_segmented(
    __global const real1* G_m0,
    __global const real1* G_m1,
    __global const real1* G_m2,
    __global const real1* G_m3,
    __global const uint* theta0,
    __global const uint* theta1,
    __global const uint* theta2,
    __global const uint* theta3,
    __constant int* args,               // args[0]=n, args[1]=k, args[2]=combo_count, args[3]=segment_size
    __global real1* max_energy_ptr,
    __global int* max_index_ptr,
    __local real1* loc_energy,
    __local int* loc_index
) {
    __global const real1* G_m[4] = { G_m0, G_m1, G_m2, G_m3 };
    __global const uint* theta[4] = { theta0, theta1, theta2, theta3 };

    const int n = args[0];
    const int shots = args[1];
    const bool is_spin_glass = args[2];
    const int segment_size = args[3];
    const int theta_segment_size = args[4];
    const int n32 = (n + 31) >> 5U;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < shots; i += max_i) {
        const int j = i * n32;
        const real1 energy = cut_worker_segmented(theta[j / theta_segment_size] + (j % theta_segment_size), G_m, n, segment_size, is_spin_glass);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 single_bit_flip_worker_segmented(__constant uint* theta, __global const real1** G_m, const int n, const int segment_size, const bool is_spin_glass, const int k) {
    const size_t k_offset = k * (size_t)n;
    const bool k_bit = !get_const_bit(theta, k);
    real1 energy = ZERO_R1;
    for (int v = 0; v < k; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = get_G_m(G_m, k_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
    }
    for (int v = k + 1; v < n; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = get_G_m(G_m, k_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
    }

    return energy;
}

__kernel void single_bit_flips_segmented(
    __global const real1* G_m0,
    __global const real1* G_m1,
    __global const real1* G_m2,
    __global const real1* G_m3,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    __global const real1* G_m[4] = { G_m0, G_m1, G_m2, G_m3 };

    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int segment_size = args[2];

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < n; i += max_i) {
        const real1 energy = single_bit_flip_worker_segmented(best_theta, G_m, n, segment_size, is_spin_glass, i);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 double_bit_flip_worker_segmented(__constant uint* theta, __global const real1** G_m, const int n, const int segment_size, const bool is_spin_glass, int k, int l) {
    if (l < k) {
        int t = k;
        k = l;
        l = t;
    }
    const size_t k_offset = k * (size_t)n;
    const bool k_bit = !get_const_bit(theta, k);
    const size_t l_offset = l * (size_t)n;
    const bool l_bit = !get_const_bit(theta, l);
    real1 energy = ZERO_R1;
    for (int v = 0; v < k; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = get_G_m(G_m, k_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
        val = get_G_m(G_m, l_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (l_bit != v_bit) ? val : -val;
    }
    for (int v = k + 1; v < l; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = get_G_m(G_m, k_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
        val = get_G_m(G_m, l_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (l_bit != v_bit) ? val : -val;
    }
    for (int v = l + 1; v < n; ++v) {
        const bool v_bit = get_const_bit(theta, v);
        real1 val = get_G_m(G_m, k_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (k_bit != v_bit) ? val : -val;
        val = get_G_m(G_m, l_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        energy += (l_bit != v_bit) ? val : -val;
    }

    return energy;
}

__kernel void double_bit_flips_segmented(
    __global const real1* G_m0,
    __global const real1* G_m1,
    __global const real1* G_m2,
    __global const real1* G_m3,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k
    __global real1* max_energy_ptr,     // output: per-group min energy
    __global int* max_index_ptr,        // output: per-group best index (i)
    __local real1* loc_energy,          // local memory buffer
    __local int* loc_index              // local memory buffer
) {
    __global const real1* G_m[4] = { G_m0, G_m1, G_m2, G_m3 };

    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int segment_size = args[2];
    const int combo_count = (n * (n - 1)) >> 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < combo_count; i += max_i) {
        int c = i;
        int k = 0;
        int lcv = n - 1;
        while (c >= lcv) {
            c -= lcv;
            ++k;
            --lcv;

            if (!lcv) {
                break;
            }
        }
        const int l = c + k + 1;

        const real1 energy = double_bit_flip_worker_segmented(best_theta, G_m, n, segment_size, is_spin_glass, k, l);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 cut_worker_sparse_segmented(
    __global const uint* theta,
    __global const real1** G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    const int n,
    const int segment_size,
    const bool is_spin_glass
) {
    real1 energy = ZERO_R1;

    for (int u = 0; u < n; ++u) {
        const bool u_bit = get_bit(theta, u);
        const uint row_end = G_rows[u + 1];
        for (uint col = G_rows[u]; col < row_end; ++col) {
            const int v = G_cols[col];
            const real1 val = get_G_m(G_data, col, segment_size);
            const bool v_bit = get_bit(theta, v);
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void calculate_cut_sparse_segmented(
    __global const real1* G_data0,
    __global const real1* G_data1,
    __global const real1* G_data2,
    __global const real1* G_data3,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __global const uint* theta0,
    __global const uint* theta1,
    __global const uint* theta2,
    __global const uint* theta3,
    __constant int* args,               // args[0] = n, args[1] = k, args[2] = combo_count, args[3] = segment_size
    __global real1* max_energy_ptr,
    __global int* max_index_ptr,
    __local real1* loc_energy,
    __local int* loc_index
) {
    __global const real1* G_data[4] = { G_data0, G_data1, G_data2, G_data3 };
    __global const uint* theta[4] = { theta0, theta1, theta2, theta3 };

    const int n = args[0];
    const int shots = args[1];
    const bool is_spin_glass = args[2];
    const int segment_size = args[3];
    const int theta_segment_size = args[4];
    const int n32 = (n + 31) >> 5U;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < shots; i += max_i) {
        const int j = i * n32;
        const real1 energy = cut_worker_sparse_segmented(theta[j / theta_segment_size] + (j % theta_segment_size), G_data, G_rows, G_cols, n, segment_size, is_spin_glass);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 single_bit_flip_worker_sparse_segmented(
    __constant uint* theta,
    __global const real1** G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    const int n,
    const int segment_size,
    const bool is_spin_glass,
    const int k
) {
    real1 energy = ZERO_R1;

    for (int u = 0; u < n; ++u) {
        bool u_bit = get_const_bit(theta, u);
        if (u == k) {
            u_bit = !u_bit;
        }
        const uint row_end = G_rows[u + 1];
        for (uint col = G_rows[u]; col < row_end; ++col) {
            const int v = G_cols[col];
            const real1 val = get_G_m(G_data, col, segment_size);
            bool v_bit = get_const_bit(theta, v);
            if (v == k) {
                v_bit = !v_bit;
            }
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void single_bit_flips_sparse_segmented(
    __global const real1* G_data0,
    __global const real1* G_data1,
    __global const real1* G_data2,
    __global const real1* G_data3,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k, args[2] = combo_count, args[3] = segment_size
    __global real1* max_energy_ptr,
    __global int* max_index_ptr,
    __local real1* loc_energy,
    __local int* loc_index
) {
    __global const real1* G_data[4] = { G_data0, G_data1, G_data2, G_data3 };

    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int segment_size = args[2];

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < n; i += max_i) {
        const real1 energy = single_bit_flip_worker_sparse_segmented(best_theta, G_data, G_rows, G_cols, n, segment_size, is_spin_glass, i);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

real1 double_bit_flip_worker_sparse_segmented(
    __constant uint* theta,
    __global const real1** G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    const int n,
    const int segment_size,
    const bool is_spin_glass,
    const int k,
    const int l
) {
    real1 energy = ZERO_R1;

    for (int u = 0; u < n; ++u) {
        bool u_bit = get_const_bit(theta, u);
        if ((u == k) || (u == l)) {
            u_bit = !u_bit;
        }
        const uint row_end = G_rows[u + 1];
        for (uint col = G_rows[u]; col < row_end; ++col) {
            const int v = G_cols[col];
            const real1 val = get_G_m(G_data, col, segment_size);
            bool v_bit = get_const_bit(theta, v);
            if ((v == k) || (v == l)) {
                v_bit = !v_bit;
            }
            if (u_bit != v_bit) {
                energy += val;
            } else if (is_spin_glass) {
                energy -= val;
            }
        }
    }

    return energy;
}

__kernel void double_bit_flips_sparse_segmented(
    __global const real1* G_data0,
    __global const real1* G_data1,
    __global const real1* G_data2,
    __global const real1* G_data3,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant uint* best_theta,
    __constant int* args,               // args[0] = n, args[1] = k, args[2] = combo_count, args[3] = segment_size
    __global real1* max_energy_ptr,
    __global int* max_index_ptr,
    __local real1* loc_energy,
    __local int* loc_index
) {
    __global const real1* G_data[4] = { G_data0, G_data1, G_data2, G_data3 };

    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int segment_size = args[2];
    const int combo_count = (n * (n - 1)) >> 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    real1 best_energy = -INFINITY;
    int best_i = i;

    for (; i < combo_count; i += max_i) {
        int c = i;
        int k = 0;
        int lcv = n - 1;
        while (c >= lcv) {
            c -= lcv;
            ++k;
            --lcv;

            if (!lcv) {
                break;
            }
        }
        const int l = c + k + 1;

        const real1 energy = double_bit_flip_worker_sparse_segmented(best_theta, G_data, G_rows, G_cols, n, segment_size, is_spin_glass, k, l);
        if (energy > best_energy) {
            best_energy = energy;
            best_i = i;
        }
    }

    reduce_energy_index(best_energy, best_i, loc_energy, loc_index, max_energy_ptr, max_index_ptr);
}

void reduce_energy_index_block(real1 energy, int i, int block, __local real1* loc_energy, __local int* loc_index, __local int* loc_block, __global real1* max_energy_ptr, __global int* max_index_ptr, __global int* max_block_ptr) {
    const int lt_id = get_local_id(0);
    const int lt_size = get_local_size(0);

    loc_energy[lt_id] = energy;
    loc_index[lt_id] = i;
    loc_block[lt_id] = block;

    // Reduce within workgroup
    for (int offset = lt_size >> 1; offset > 0; offset >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        real1 hid_energy, lid_energy;
        if (lt_id < offset) {
            hid_energy = loc_energy[lt_id + offset];
            lid_energy = loc_energy[lt_id];
            if (hid_energy > lid_energy) {
                loc_energy[lt_id] = hid_energy;
                loc_index[lt_id] = loc_index[lt_id + offset];
                loc_block[lt_id] = loc_block[lt_id + offset];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Write out per-group result
    if (lt_id == 0) {
        max_energy_ptr[get_group_id(0)] = loc_energy[0];
        max_index_ptr[get_group_id(0)] = loc_index[0];
        max_block_ptr[get_group_id(0)] = loc_block[0];
    }
}

inline bool get_local_bit(ulong* theta, const size_t u) {
    return (theta[u >> 6U] >> (u & 63U)) & 1U;
}

inline bool get_const_long_bit(__constant ulong* theta, const size_t u) {
    return (theta[u >> 6U] >> (u & 63U)) & 1U;
}

inline size_t gray_code_next(ulong* theta, const size_t curr_idx, const size_t offset) {
    size_t prev = curr_idx;
    size_t curr = curr_idx + 1U;
    prev = prev ^ (prev >> 1U);
    curr = curr ^ (curr >> 1U);
    size_t diff = prev ^ curr;
    size_t flip_bit = 0;
    while (!((diff >> flip_bit) & 1U)) {
        ++flip_bit;
    }
    flip_bit += offset;
    theta[flip_bit >> 6U] ^= 1UL << (flip_bit & 63U);

    return flip_bit;
}


__kernel void gray(
    __global const real1* G_m,
    __constant ulong* theta,
    __constant int* args,
    __global ulong* theta_out,
    __global real1* energy_out
) {
    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int gray_iterations = args[2];
    const int blocks = (n + 63) >> 6U;
    const int last_block = blocks - 1;
    const int rem = n - (blocks << 6U);

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    ulong theta_local[2048];
    for (int b = 0; b < blocks; ++b) {
        theta_local[b] = theta[b];
    }

    // Initialize different seed per thread
    const ulong seed = i ^ (i >> 1);
    ulong diff_mask = 0U;
    const int max_lcv = (n < 64) ? n : 64;
    for (int b = 0; b < max_lcv; ++b) {
        const bool bit = (seed >> (63U - b)) & 1U;
        if (!bit) {
            continue;
        }
        const int offset = n - (b + 1);
        const int bit_offset = offset & 63U;
        const ulong p = 1 << bit_offset;
        diff_mask |= p;
        theta_local[offset >> 6U] ^= p;
    }

    real1 best_energy = ZERO_R1;
    for (int b = 0; b < max_lcv; ++b) {
        const int u = b + rem;
        const size_t u_offset = u * n;
        if (!((diff_mask >> b) & 1UL)) {
            continue;
        }
        const bool n_bit = get_local_bit(theta_local, u);
        for (uint v = 0; v < u; ++v) {
            const bool v_bit = get_const_long_bit(theta, v);
            real1 val = G_m[u_offset + v];
            best_energy += (n_bit == v_bit) ? -val : val;
        }
        for (uint v = u + 1; v < n; ++v) {
            bool v_bit = get_const_long_bit(theta, v);
            real1 val = G_m[u_offset + v];
            best_energy += (n_bit == v_bit) ? -val : val;
        }
    }

    if (!is_spin_glass) {
        best_energy *= -ONE_R1 / 2;
    }

    for (; i < gray_iterations; i += max_i) {
        for (int block = 0; block < blocks; ++block) {
            const size_t k = gray_code_next(theta_local, i, block << 6U);
            const size_t k_offset = k * n;
            const bool k_bit = get_local_bit(theta_local, k);

            real1 energy = ZERO_R1;
            for (uint v = 0; v < k; v++) {
                const bool v_bit = get_local_bit(theta_local, v);
                real1 val = G_m[k_offset + v];
                if (is_spin_glass) {
                    val *= 2;
                }
                energy += (k_bit != v_bit) ? val : -val;
            }
            for (uint v = k + 1; v < n; v++) {
                const bool v_bit = get_local_bit(theta_local, v);
                real1 val = G_m[k_offset + v];
                if (is_spin_glass) {
                    val *= 2;
                }
                energy += (k_bit != v_bit) ? val : -val;
            }

            if (energy > ZERO_R1) {
                best_energy += energy;
            } else {
                theta_local[k >> 6U] ^= 1UL << (k & 63U);
            }
        }
    }

    i = get_global_id(0);
    const size_t offset = i * blocks;
    for (int b = 0; b < blocks; ++b) {
        theta_out[offset + b] = theta_local[b];
    }
    energy_out[i] = best_energy;
}

__kernel void gray_segmented(
    __global const real1* G_m0,
    __global const real1* G_m1,
    __global const real1* G_m2,
    __global const real1* G_m3,
    __constant ulong* theta,
    __constant int* args,
    __global ulong* theta_out,
    __global real1* energy_out
) {
    __global const real1* G_m[4] = { G_m0, G_m1, G_m2, G_m3 };

    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int gray_iterations = args[2];
    const int segment_size = args[3];
    const int blocks = (n + 63) / 64;
    const int last_block = blocks - 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    ulong theta_local[2048];
    for (int b = 0; b < blocks; ++b) {
        theta_local[b] = theta[b];
    }

    // Initialize different seed per thread
    const ulong seed = i ^ (i >> 1);
    int b;
    for (b = 0; b < 64; ++b) {
        if (seed >> (63U - b)) {
            break;
        }
    }
    theta_local[last_block] ^= (seed >> (63U - b)) << b;

    const size_t b_offset = b * n;
    const bool b_bit = get_local_bit(theta_local, b);

    real1 best_energy = ZERO_R1;
    for (uint v = 0; v < b; v++) {
        const bool v_bit = get_local_bit(theta_local, v);
        real1 val = get_G_m(G_m, b_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        best_energy += (b_bit != v_bit) ? val : -val;
    }
    for (uint v = b + 1; v < n; v++) {
        const bool v_bit = get_local_bit(theta_local, v);
        real1 val = get_G_m(G_m, b_offset + v, segment_size);
        if (is_spin_glass) {
            val *= 2;
        }
        best_energy += (b_bit != v_bit) ? val : -val;
    }

    for (; i < gray_iterations; i += max_i) {
        for (int block = 0; block < blocks; ++block) {
            const size_t k = gray_code_next(theta_local, i, block << 6U);
            const size_t k_offset = k * n;
            const bool k_bit = get_local_bit(theta_local, k);

            real1 energy = ZERO_R1;
            for (uint v = 0; v < k; v++) {
                const bool v_bit = get_local_bit(theta_local, v);
                real1 val = get_G_m(G_m, k_offset + v, segment_size);
                if (is_spin_glass) {
                    val *= 2;
                }
                energy += (k_bit != v_bit) ? val : -val;
            }
            for (uint v = k + 1; v < n; v++) {
                const bool v_bit = get_local_bit(theta_local, v);
                real1 val = get_G_m(G_m, k_offset + v, segment_size);
                if (is_spin_glass) {
                    val *= 2;
                }
                energy += (k_bit != v_bit) ? val : -val;
            }

            if (energy > ZERO_R1) {
                best_energy += energy;
            } else {
                theta_local[k >> 6U] ^= 1UL << (k & 63U);
            }
        }
    }

    i = get_global_id(0);
    const size_t offset = i * blocks;
    for (int b = 0; b < blocks; ++b) {
        theta_out[offset + b] = theta_local[b];
    }
    energy_out[i] = best_energy;
}

__kernel void gray_sparse(
    __global const real1* G_data,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant ulong* theta,
    __constant int* args,
    __global ulong* theta_out,
    __global real1* energy_out
) {
    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int gray_iterations = args[2];
    const int blocks = (n + 63) / 64;
    const int last_block = blocks - 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    ulong theta_local[2048];
    for (int b = 0; b < blocks; ++b) {
        theta_local[b] = theta[b];
    }

    // Initialize different seed per thread
    const ulong seed = i ^ (i >> 1);
    for (int b = 0; b < 64; ++b) {
        theta_local[last_block] ^= (seed >> (63U - b)) << b;
    }

    real1 best_energy = ZERO_R1;
    for (uint u = 0; u < n; u++) {
        const bool u_bit = get_local_bit(theta_local, u);
        const size_t mCol = G_rows[u + 1];
        for (int col = G_rows[u]; col < mCol; ++col) {
            const int v = G_cols[col];
            const real1 val = G_data[col];
            const bool v_bit = get_local_bit(theta_local, v);
            if (u_bit != v_bit) {
                best_energy += val;
            } else if (is_spin_glass) {
                best_energy -= val;
            }
        }
    }

    for (; i < gray_iterations; i += max_i) {
        for (int block = 0; block < blocks; ++block) {
            const size_t flip_bit = gray_code_next(theta_local, i, block << 6U);
            real1 energy = ZERO_R1;
            for (uint u = 0; u < n; u++) {
                const bool u_bit = get_local_bit(theta_local, u);
                const size_t mCol = G_rows[u + 1];
                for (int col = G_rows[u]; col < mCol; ++col) {
                    const int v = G_cols[col];
                    const real1 val = G_data[col];
                    const bool v_bit = get_local_bit(theta_local, v);
                    if (u_bit != v_bit) {
                        energy += val;
                    } else if (is_spin_glass) {
                        energy -= val;
                    }
                }
            }

            if (energy > best_energy) {
                best_energy = energy;
            } else {
                theta_local[flip_bit >> 6U] ^= 1UL << (flip_bit & 63U);
            }
        }
    }

    i = get_global_id(0);
    const size_t offset = i * blocks;
    for (int b = 0; b < blocks; ++b) {
        theta_out[offset + b] = theta_local[b];
    }
    energy_out[i] = best_energy;
}

__kernel void gray_sparse_segmented(
    __global const real1* G_data0,
    __global const real1* G_data1,
    __global const real1* G_data2,
    __global const real1* G_data3,
    __global const uint* G_rows,
    __global const uint* G_cols,
    __constant ulong* theta,
    __constant int* args,
    __global ulong* theta_out,
    __global real1* energy_out
) {
    __global const real1* G_data[4] = { G_data0, G_data1, G_data2, G_data3 };

    const int n = args[0];
    const bool is_spin_glass = args[1];
    const int gray_iterations = args[2];
    const int segment_size = args[3];
    const int blocks = (n + 63) / 64;
    const int last_block = blocks - 1;

    int i = get_global_id(0);
    const int max_i = get_global_size(0);

    ulong theta_local[2048];
    for (int b = 0; b < blocks; ++b) {
        theta_local[b] = theta[b];
    }

    // Initialize different seed per thread
    const ulong seed = i ^ (i >> 1);
    for (int b = 0; b < 64; ++b) {
        theta_local[last_block] ^= (seed >> (63U - b)) << b;
    }

    real1 best_energy = ZERO_R1;
    for (uint u = 0; u < n; u++) {
        const size_t u_offset = u * n;
        const bool u_bit = get_local_bit(theta_local, u);
        const uint row_end = G_rows[u + 1];
        for (uint col = G_rows[u]; col < row_end; ++col) {
            const int v = G_cols[col];
            const real1 val = get_G_m(G_data, col, segment_size);
            const bool v_bit = get_local_bit(theta_local, v);
            if (u_bit != v_bit) {
                best_energy += val;
            } else if (is_spin_glass) {
                best_energy -= val;
            }
        }
    }

    for (; i < gray_iterations; i += max_i) {
        for (int block = 0; block < blocks; ++block) {
            const size_t flip_bit = gray_code_next(theta_local, i, block << 6U);
            real1 energy = ZERO_R1;
            for (uint u = 0; u < n; u++) {
                const size_t u_offset = u * n;
                const bool u_bit = get_local_bit(theta_local, u);
                const uint row_end = G_rows[u + 1];
                for (uint col = G_rows[u]; col < row_end; ++col) {
                    const int v = G_cols[col];
                    const real1 val = get_G_m(G_data, col, segment_size);
                    const bool v_bit = get_local_bit(theta_local, v);
                    if (u_bit != v_bit) {
                        energy += val;
                    } else if (is_spin_glass) {
                        energy -= val;
                    }
                }
            }

            if (energy > best_energy) {
                best_energy = energy;
            } else {
                theta_local[flip_bit >> 6U] ^= 1UL << (flip_bit & 63U);
            }
        }
    }

    i = get_global_id(0);
    const size_t offset = i * blocks;
    for (int b = 0; b < blocks; ++b) {
        theta_out[offset + b] = theta_local[b];
    }
    energy_out[i] = best_energy;
}
