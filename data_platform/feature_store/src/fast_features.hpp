#pragma once
#include <cmath>

class FastFeatures {
public:
    static void compute_sma(const double* prices, size_t n, size_t window, double* out) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::nan("");
        }
        if (n < window || window == 0) return;

        double sum = 0.0;
        for (size_t i = 0; i < window; ++i) {
            sum += prices[i];
        }
        out[window - 1] = sum / window;

        for (size_t i = window; i < n; ++i) {
            sum += prices[i] - prices[i - window];
            out[i] = sum / window;
        }
    }

    static void compute_std_dev(const double* prices, size_t n, size_t window, double* out) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::nan("");
        }
        if (n < window || window < 2) return;

        double sum = 0.0;
        double sum_sq = 0.0;
        for (size_t i = 0; i < window; ++i) {
            sum += prices[i];
            sum_sq += prices[i] * prices[i];
        }
        
        auto get_std = [](double s, double s_sq, size_t w) {
            double mean = s / w;
            double variance = (s_sq / w) - (mean * mean);
            return variance > 0.0 ? std::sqrt(variance) : 0.0;
        };

        out[window - 1] = get_std(sum, sum_sq, window);

        for (size_t i = window; i < n; ++i) {
            sum += prices[i] - prices[i - window];
            sum_sq += prices[i] * prices[i] - prices[i - window] * prices[i - window];
            out[i] = get_std(sum, sum_sq, window);
        }
    }

    static void compute_vwap(const double* prices, const double* volumes, size_t n, size_t window, double* out) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = std::nan("");
        }
        if (n < window || window == 0) return;

        double pv_sum = 0.0;
        double vol_sum = 0.0;
        for (size_t i = 0; i < window; ++i) {
            pv_sum += prices[i] * volumes[i];
            vol_sum += volumes[i];
        }
        out[window - 1] = vol_sum > 0.0 ? pv_sum / vol_sum : 0.0;

        for (size_t i = window; i < n; ++i) {
            pv_sum += prices[i] * volumes[i] - prices[i - window] * volumes[i - window];
            vol_sum += volumes[i] - volumes[i - window];
            out[i] = vol_sum > 0.0 ? pv_sum / vol_sum : 0.0;
        }
    }
};
