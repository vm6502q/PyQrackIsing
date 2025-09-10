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
    float delta_t = args[0];
    float tot_t = args[1];
    float h_mult = args[2];
    int step = get_group_id(0) / n_qubits;
    int qi = get_group_id(0) % n_qubits;
    float J_eff = J_func[qi];
    float z = degrees[qi];
    if (fabs(z * J_eff) <= (pow(2.0f, -52))) return;

    float theta_eff = theta[qi];
    float t = step * delta_t;
    float tm1 = (step - 1) * delta_t;
    float h_t = h_mult * (tot_t - t);

    int qo = get_local_id(0);
    int n_threads = get_local_size(0);

    while (qo < n_qubits) {
        int _qo = (J_eff > 0.0f) ? (n_qubits - (1 + qo)) : qo;
        float diff = probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, t, n_qubits);
        diff -= probability_by_hamming_weight(_qo, J_eff, h_t, z, theta_eff, tm1, n_qubits);
        AtomicAdd_g_f(&(hamming_prob[_qo]), diff);
        qo += n_threads;
    }
}

