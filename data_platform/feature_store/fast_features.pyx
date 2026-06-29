# distutils: language = c++
# cython: language_level = 3

import numpy as np
cimport numpy as cn

cdef extern from "src/fast_features.hpp":
    cdef cppclass FastFeatures:
        @staticmethod
        void compute_sma(const double* prices, size_t n, size_t window, double* out)
        @staticmethod
        void compute_std_dev(const double* prices, size_t n, size_t window, double* out)
        @staticmethod
        void compute_vwap(const double* prices, const double* volumes, size_t n, size_t window, double* out)

def sma(cn.ndarray[double, ndim=1] prices, size_t window):
    cdef size_t n = prices.shape[0]
    cdef cn.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    if n > 0:
        FastFeatures.compute_sma(&prices[0], n, window, &out[0])
    return out

def std_dev(cn.ndarray[double, ndim=1] prices, size_t window):
    cdef size_t n = prices.shape[0]
    cdef cn.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    if n > 0:
        FastFeatures.compute_std_dev(&prices[0], n, window, &out[0])
    return out

def vwap(cn.ndarray[double, ndim=1] prices, cn.ndarray[double, ndim=1] volumes, size_t window):
    cdef size_t n = prices.shape[0]
    cdef cn.ndarray[double, ndim=1] out = np.empty(n, dtype=np.float64)
    if n > 0 and volumes.shape[0] == n:
        FastFeatures.compute_vwap(&prices[0], &volumes[0], n, window, &out[0])
    return out
