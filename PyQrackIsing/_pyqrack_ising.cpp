// C++/pybind11 translation of TFIM function and maxcut_tfim()
// Original algorithm by Dan Strano and Elara (OpenAI GPT), 2024

#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <bitset>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <tuple>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/functional/hash.hpp>

namespace py = pybind11;

typedef boost::multiprecision::cpp_int BigInteger;

std::random_device rd = std::random_device{};
std::mt19937 rng(rd());


static inline std::vector<double> probability_by_hamming_weight(double J, double h, double z, double theta, double t, size_t n_qubits)
{
    // critical angle
    const double theta_c = std::asin(std::max(-1.0, std::min(1.0, (std::abs(z * J) >= (std::numeric_limits<double>::epsilon() / 2)) ? std::abs(h) / (z * J) : (J > 0 ? 1.0 : -1.0))));
    const double delta_theta = theta - theta_c;
    std::vector<double> bias(n_qubits + 1, 0.0);
    if (std::abs(h) < 1e-12) {
        bias[0] = 1.0;
    } else if (std::abs(J) < 1e-12) {
        std::fill(bias.begin(), bias.end(), 1.0 / (n_qubits + 1.0));
    } else {
        const double sin_delta = std::sin(delta_theta);
        const double omega = 1.5 * M_PI;
        const double t2 = 1.0;
        const double p = std::pow(2.0, std::abs(J / h) - 1.0) * (1.0 + sin_delta * std::cos(J * omega * t + theta) / (1.0 + std::sqrt(t / t2))) - 0.5;
        if (p >= 1024) {
            bias[0] = 1.0;
        } else {
            double tot_n = 0.0;
            for (size_t q = 0U; q <= n_qubits; ++q) {
                double n = 1.0 / ((n_qubits + 1) * std::pow(2.0, p * q));
                bias[q] = n;
                tot_n += n;
            }
            for (size_t q = 0U; q <= n_qubits; ++q) {
                bias[q] /= tot_n;
            }
        }
    }
    if (J > 0) {
        std::reverse(bias.begin(), bias.end());
    }

    return bias;
}

static inline std::string int_to_bitstring(BigInteger integer, size_t length) {
    std::string s(length, '0');
    for (size_t i = 0; i < length; ++i) {
        if (integer & (1ULL << (length - 1U - i))) {
            s[i] = '1';
        }
    }
    return s;
}

std::vector<double> maxcut_hamming_cdf(size_t n_qubits, std::vector<double> J_func, std::vector<double> degrees, int mult_log2) {
    if (!n_qubits) {
        return std::vector<double>();
    }

    const size_t n_steps = n_qubits << mult_log2;
    const size_t shots = n_qubits << mult_log2;
    const double delta_t = 1.0 / (n_steps << (mult_log2 >> 1));
    const double h_mult = (1 << (mult_log2 >> 1)) / (n_steps * delta_t);
    std::vector<double> hamming_prob(n_qubits - 1U, 0.0);

    for (size_t step = 0; step < n_steps; ++step) {
        double t = step * delta_t;
        double tm1 = (step - 1) * delta_t;
        for (size_t q = 0; q < n_qubits; ++q) {
            const double& z = degrees[q];
            const double J_eff = J_func[q];
            const double h_t = h_mult * t;
            auto bias = probability_by_hamming_weight(J_eff, h_t, z, 0.0, t, n_qubits);
            if (step == 0) {
                for (size_t i = 0U; i < hamming_prob.size(); ++i) {
                    hamming_prob[i] += bias[i + 1U];
                }
                continue;
            }
            auto last_bias = probability_by_hamming_weight(J_eff, h_t, z, 0.0, tm1, n_qubits);
            for (size_t i = 0U; i < hamming_prob.size(); ++i) {
                hamming_prob[i] += bias[i + 1U] - last_bias[i + 1U];
            }
        }
    }

    double tot_prob = std::accumulate(hamming_prob.begin(), hamming_prob.end(), 0.0);
    for (auto& x : hamming_prob) {
        x /= tot_prob;
    }

    tot_prob = 0.0;
    for (size_t i = 0U; i < hamming_prob.size(); ++i) {
        tot_prob += hamming_prob[i];
        hamming_prob[i] = tot_prob;
    }
    hamming_prob.back() = 1.0;

    return hamming_prob;
}

static inline double closeness_like_bits(BigInteger perm, size_t n_rows, size_t n_cols) {
    std::string bits = int_to_bitstring(perm, n_rows * n_cols);
    double like_count = 0.0;
    double total_edges = 0.0;
    // grid neighbors (right and down)
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            char s = bits[i * n_cols + j];
            char s_right = bits[i * n_cols + ((j + 1) % n_cols)];
            char s_down = bits[((i + 1) % n_rows) * n_cols + j];
            like_count += (s == s_right) ? 1.0 : -1.0;
            like_count += (s == s_down) ? 1.0 : -1.0;
            total_edges += 2.0;
        }
    }
    return like_count / total_edges;
}

static inline double expected_closeness_weight(size_t n_rows, size_t n_cols, size_t hamming_weight) {
    const size_t L = n_rows * n_cols;
    auto comb = [](size_t n, size_t k) {
        if ((k < 0) || (k > n)) {
            return 0ULL;
        }
        if ((k == 0) || (k == n)) {
            return 1ULL;
        }
        unsigned long long res = 1ULL;
        for (size_t i = 1; i <= k; ++i) {
            res = res * (n - k + i) / i;
        }
        return res;
    };
    const double same_pairs = comb(hamming_weight, 2U) + comb(L - hamming_weight, 2U);
    const double total_pairs = comb(L, 2U);
    const double mu_k = same_pairs / total_pairs;

    return 2.0 * mu_k - 1.0;
}

std::vector<std::string> generate_tfim_samples_cpp(double J, double h, double z, double theta, double t, size_t n_qubits, size_t shots) {
    // lattice dimensions
    size_t n_rows = 1U;
    size_t n_cols = n_qubits;
    for (size_t c = std::floor(std::sqrt(n_qubits)); c >= 1; --c) {
        if ((n_qubits % c) == 0U) {
            n_rows = n_qubits / c;
            n_cols = c;
            break;
        }
    }

    const std::vector<double> bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits);

    // thresholds
    std::vector<double> thresholds(n_qubits + 1);
    double tot_prob = 0.0;
    for (size_t q = 0U; q <= n_qubits; ++q) {
        tot_prob += bias[q];
        thresholds[q] = tot_prob;
    }
    thresholds[n_qubits] = 1.0;

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<BigInteger> samples;
    samples.reserve(shots);

    std::vector<int> qubits(n_qubits);
    for (size_t i = 0U; i < n_qubits; ++i) {
        qubits[i] = i;
    }

    std::vector<size_t> hamming_samples(n_qubits + 1U);
    for (size_t s = 0U; s < shots; ++s) {
        double mag_prob = dist(rng);
        size_t m = 0U;
        while (thresholds[m] < mag_prob) {
            ++m;
        }
        ++hamming_samples[m];
    }
    for (size_t m = 0U; m < hamming_samples.size(); ++m){
        double tot_cprob = 0.0;
        size_t hs = hamming_samples[m];
        std::vector<double> rands;
        rands.reserve(hs);
        for (size_t s = 0U; s < hs; ++s) {
            rands.push_back(dist(rng));
        }
        std::sort(rands.begin(), rands.end());
        // iterate combinations
        std::vector<size_t> idx(m);
        for (size_t i = 0U; i < m; ++i) {
            idx[i] = i;
        }
        size_t s = 0U;
        while (true) {
            BigInteger candidate = 0U;
            for (size_t pos : idx) {
                candidate |= (BigInteger("1") << pos);
            }
            tot_cprob += (1.0 + closeness_like_bits(candidate, n_rows, n_cols)) /
                         (1.0 + expected_closeness_weight(n_rows, n_cols, m));
            while ((s < hs) && (rands[s] <= tot_cprob)) {
                samples.push_back(candidate);
                ++s;
            }
            if (s == hs) {
                break;
            }

            // next combination
            int64_t k = m - 1;
            while ((k >= 0) && (idx[k] == (n_qubits - m + k))) {
                --k;
            }
            if (k < 0) {
                for (; s < hs; ++s) {
                    samples.push_back(candidate);
                }
                break;
            }
            ++idx[k];
            for (int64_t j = k + 1; j < m; ++j) {
                idx[j] = idx[j - 1U] + 1;
            }
        }
    }

    auto dre = std::default_random_engine{rd()};
    std::shuffle(samples.begin(), samples.end(), dre);

    std::vector<std::string> output;
    output.reserve(shots);
    for (BigInteger& s : samples) {
        output.push_back(boost::lexical_cast<std::string>(s));
    }

    return output;
}

double tfim_magnetization(double J, double h, double z, double theta, double t, size_t n_qubits) {
    const std::vector<double> bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits);
    double magnetization = 0.0;
    const int64_t nqs = (int64_t)n_qubits;
    const double nqd = (double)n_qubits;
    for (int64_t q = 0U; q < bias.size(); ++q) {
        const double mag = (nqs - 2 * q) / nqd;
        magnetization += bias[q] * mag;
    }

    return magnetization;
}

double tfim_square_magnetization(double J, double h, double z, double theta, double t, size_t n_qubits) {
    const std::vector<double> bias = probability_by_hamming_weight(J, h, z, theta, t, n_qubits);
    double square_magnetization = 0.0;
    const int64_t nqs = (int64_t)n_qubits;
    const double nqd = (double)n_qubits;
    for (int64_t q = 0U; q < bias.size(); ++q) {
        const double mag = (nqs - 2 * q) / nqd;
        square_magnetization += bias[q] * mag * mag;
    }

    return square_magnetization;
}

PYBIND11_MODULE(tfim_sampler, m) {
    m.doc() = "PyQrackIsing TFIM sample generator";
    m.def("_generate_tfim_samples", &generate_tfim_samples_cpp,
          py::arg("J"), py::arg("h"), py::arg("z"), py::arg("theta"),
          py::arg("t"), py::arg("n_qubits"), py::arg("shots"));
    m.def("_tfim_magnetization", &tfim_magnetization,
          py::arg("J"), py::arg("h"), py::arg("z"), py::arg("theta"),
          py::arg("t"), py::arg("n_qubits"));
    m.def("_tfim_square_magnetization", &tfim_square_magnetization,
          py::arg("J"), py::arg("h"), py::arg("z"), py::arg("theta"),
          py::arg("t"), py::arg("n_qubits"));
    m.def("_maxcut_hamming_cdf", &maxcut_hamming_cdf, "Adiabatic TFIM Hamming weight probability density function");
}

