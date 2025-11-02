/*
 * Copyright 2021 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file cholesky.hpp
 * @brief This file contains cholesky functions
 *   - cholesky                 : Entry point function
 *   - choleskyTop             : Top level function that selects implementation architecture and internal types based
 * on a traits class.
 *   - choleskyBasic           : Basic implementation requiring lower resource
 *   - choleskyAlt             : Lower latency architecture requiring more resources
 *   - choleskyAlt2            : Further improved latency architecture requiring higher resource
 */

#ifndef _XF_SOLVER_CHOLESKY_HPP_
#define _XF_SOLVER_CHOLESKY_HPP_

#include "ap_fixed.h"
#include "hls_x_complex.h"
#include "hls_math.h"
#include <complex>
#include "utils/std_complex_utils.h"
#include "utils/x_matrix_utils.hpp"
#include "hls_stream.h"

namespace xf {
namespace solver {

namespace details {
inline float fast_inv_sqrt_float(float x) {
#pragma HLS INLINE
    return x_rsqrt(x);
}
// Derive the preferred architecture/unroll strategy from matrix size.
template <int RowsColsA>
struct cholesky_opt_config {
    static const int SAFE_DIM = (RowsColsA > 0) ? RowsColsA : 1;
    static const bool USE_SMALL_CORE = (SAFE_DIM <= 4);
    static const int ARCH = (SAFE_DIM <= 4 ? 1 : 2);
    static const int UNROLL = (SAFE_DIM <= 4 ? SAFE_DIM : (SAFE_DIM <= 8 ? 8 : (SAFE_DIM <= 16 ? 4 : 2)));
    static const bool ARCH2_ZERO_LOOP = false;  // Disable separate zero loop to merge zeroing into main loop
};

template <typename T, typename Scalar>
inline void set_real_zero_imag(T& dst, const Scalar& value) {
    dst = static_cast<T>(value);
}

template <typename Scalar>
inline void set_real_zero_imag(hls::x_complex<Scalar>& dst, const Scalar& value) {
    dst.real(value);
    dst.imag(0);
}

template <typename Scalar>
inline void set_real_zero_imag(std::complex<Scalar>& dst, const Scalar& value) {
    dst.real(value);
    dst.imag(0);
}

template <typename T>
inline T conj_copy(const T& val) {
    return val;
}

template <typename Scalar>
inline hls::x_complex<Scalar> conj_copy(const hls::x_complex<Scalar>& val) {
    return hls::x_conj(val);
}

template <typename Scalar>
inline std::complex<Scalar> conj_copy(const std::complex<Scalar>& val) {
    return std::conj(val);
}

// Compute magnitude squared |z|^2 = Re^2 + Im^2 (faster than conj(z)*z for complex)
template <typename T>
inline T magnitude_squared(const T& val) {
#pragma HLS INLINE
    return val * val;
}

template <typename Scalar>
inline Scalar magnitude_squared(const hls::x_complex<Scalar>& val) {
#pragma HLS INLINE
    Scalar re = val.real();
    Scalar im = val.imag();
    return re * re + im * im;
}

template <typename Scalar>
inline Scalar magnitude_squared(const std::complex<Scalar>& val) {
#pragma HLS INLINE
    Scalar re = val.real();
    Scalar im = val.imag();
    return re * re + im * im;
}
} // namespace details

// ===================================================================================================================
// Default traits struct defining the internal variable types for the cholesky function
template <bool LowerTriangularL, int RowsColsA, typename InputType, typename OutputType>
struct choleskyTraits {
    typedef InputType PROD_T;
    typedef InputType ACCUM_T;
    typedef InputType ADD_T;
    typedef InputType DIAG_T;
    typedef InputType RECIP_DIAG_T;
    typedef InputType OFF_DIAG_T;
    typedef OutputType L_OUTPUT_T;
    static const int ARCH = details::cholesky_opt_config<RowsColsA>::ARCH;
    static const int INNER_II = 1; // Specify the pipelining target for the inner loop
    static const int UNROLL_FACTOR =
        details::cholesky_opt_config<RowsColsA>::UNROLL; // Unroll aggressively for very small matrices
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2); // Dimension to unroll matrix
    static const bool ARCH2_ZERO_LOOP = details::cholesky_opt_config<RowsColsA>::ARCH2_ZERO_LOOP;
};

// Specialization for complex
template <bool LowerTriangularL, int RowsColsA, typename InputBaseType, typename OutputBaseType>
struct choleskyTraits<LowerTriangularL, RowsColsA, hls::x_complex<InputBaseType>, hls::x_complex<OutputBaseType> > {
    typedef hls::x_complex<InputBaseType> PROD_T;
    typedef hls::x_complex<InputBaseType> ACCUM_T;
    typedef hls::x_complex<InputBaseType> ADD_T;
    typedef hls::x_complex<InputBaseType> DIAG_T;
    typedef InputBaseType RECIP_DIAG_T;
    typedef hls::x_complex<InputBaseType> OFF_DIAG_T;
    typedef hls::x_complex<OutputBaseType> L_OUTPUT_T;
    static const int ARCH = details::cholesky_opt_config<RowsColsA>::ARCH;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = details::cholesky_opt_config<RowsColsA>::UNROLL;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const bool ARCH2_ZERO_LOOP = details::cholesky_opt_config<RowsColsA>::ARCH2_ZERO_LOOP;
};

// Specialization for std complex
template <bool LowerTriangularL, int RowsColsA, typename InputBaseType, typename OutputBaseType>
struct choleskyTraits<LowerTriangularL, RowsColsA, std::complex<InputBaseType>, std::complex<OutputBaseType> > {
    typedef std::complex<InputBaseType> PROD_T;
    typedef std::complex<InputBaseType> ACCUM_T;
    typedef std::complex<InputBaseType> ADD_T;
    typedef std::complex<InputBaseType> DIAG_T;
    typedef InputBaseType RECIP_DIAG_T;
    typedef std::complex<InputBaseType> OFF_DIAG_T;
   typedef std::complex<OutputBaseType> L_OUTPUT_T;
    static const int ARCH = details::cholesky_opt_config<RowsColsA>::ARCH;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = details::cholesky_opt_config<RowsColsA>::UNROLL;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const bool ARCH2_ZERO_LOOP = details::cholesky_opt_config<RowsColsA>::ARCH2_ZERO_LOOP;
};

// Specialization for ap_fixed
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL, RowsColsA, ap_fixed<W1, I1, Q1, O1, N1>, ap_fixed<W2, I2, Q2, O2, N2> > {
    typedef ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> PROD_T;
    typedef ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                     (I1 + I1) + BitWidth<RowsColsA>::Value,
                     AP_RND_CONV,
                     AP_SAT,
                     0>
        ACCUM_T;
    typedef ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> ADD_T;
    typedef ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> DIAG_T;     // Takes result of sqrt
    typedef ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0>
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = details::cholesky_opt_config<RowsColsA>::ARCH;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = details::cholesky_opt_config<RowsColsA>::UNROLL;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const bool ARCH2_ZERO_LOOP = details::cholesky_opt_config<RowsColsA>::ARCH2_ZERO_LOOP;
};

// Further specialization for hls::complex<ap_fixed>
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL,
                      RowsColsA,
                      hls::x_complex<ap_fixed<W1, I1, Q1, O1, N1> >,
                      hls::x_complex<ap_fixed<W2, I2, Q2, O2, N2> > > {
    // Optimized Phase 4.1: Incrementally increase DIAG_T/OFF_DIAG_T reduction to -7 bits
    // PROD_T/ACCUM_T: -4 bits, DIAG_T/OFF_DIAG_T: -7 bits (was -6), RECIP_DIAG_T: -4 bits
    typedef hls::x_complex<ap_fixed<W1 + W1 - 4, I1 + I1, AP_RND, AP_WRAP, 0> > PROD_T;
    typedef hls::x_complex<ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value - 4,
                                    (I1 + I1) + BitWidth<RowsColsA>::Value,
                                    AP_RND,
                                    AP_WRAP,
                                    0> >
        ACCUM_T;
    typedef hls::x_complex<ap_fixed<W1 + 1, I1 + 1, AP_RND, AP_WRAP, 0> > ADD_T;
    typedef hls::x_complex<ap_fixed<(W1 + 1) * 2 - 7, I1 + 1, AP_RND, AP_WRAP, 0> > DIAG_T;     // Takes result of sqrt
    typedef hls::x_complex<ap_fixed<(W1 + 1) * 2 - 7, I1 + 1, AP_RND, AP_WRAP, 0> > OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2 - 4, 2 + (W2 - I2), AP_RND, AP_WRAP, 0> RECIP_DIAG_T;
    typedef hls::x_complex<ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0> >
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = details::cholesky_opt_config<RowsColsA>::ARCH;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = details::cholesky_opt_config<RowsColsA>::UNROLL;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const bool ARCH2_ZERO_LOOP = details::cholesky_opt_config<RowsColsA>::ARCH2_ZERO_LOOP;
};

// Further specialization for std::complex<ap_fixed>
template <bool LowerTriangularL,
          int RowsColsA,
          int W1,
          int I1,
          ap_q_mode Q1,
          ap_o_mode O1,
          int N1,
          int W2,
          int I2,
          ap_q_mode Q2,
          ap_o_mode O2,
          int N2>
struct choleskyTraits<LowerTriangularL,
                      RowsColsA,
                      std::complex<ap_fixed<W1, I1, Q1, O1, N1> >,
                      std::complex<ap_fixed<W2, I2, Q2, O2, N2> > > {
    typedef std::complex<ap_fixed<W1 + W1, I1 + I1, AP_RND_CONV, AP_SAT, 0> > PROD_T;
    typedef std::complex<ap_fixed<(W1 + W1) + BitWidth<RowsColsA>::Value,
                                  (I1 + I1) + BitWidth<RowsColsA>::Value,
                                  AP_RND_CONV,
                                  AP_SAT,
                                  0> >
        ACCUM_T;
    typedef std::complex<ap_fixed<W1 + 1, I1 + 1, AP_RND_CONV, AP_SAT, 0> > ADD_T;
    typedef std::complex<ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > DIAG_T;     // Takes result of sqrt
    typedef std::complex<ap_fixed<(W1 + 1) * 2, I1 + 1, AP_RND_CONV, AP_SAT, 0> > OFF_DIAG_T; // Takes result of /
    typedef ap_fixed<2 + (W2 - I2) + W2, 2 + (W2 - I2), AP_RND_CONV, AP_SAT, 0> RECIP_DIAG_T;
    typedef std::complex<ap_fixed<W2, I2, AP_RND_CONV, AP_SAT, 0> >
        L_OUTPUT_T; // Takes new L value.  Same as L output but saturation set
    static const int ARCH = details::cholesky_opt_config<RowsColsA>::ARCH;
    static const int INNER_II = 1;
    static const int UNROLL_FACTOR = details::cholesky_opt_config<RowsColsA>::UNROLL;
    static const int UNROLL_DIM = (LowerTriangularL == true ? 1 : 2);
    static const bool ARCH2_ZERO_LOOP = details::cholesky_opt_config<RowsColsA>::ARCH2_ZERO_LOOP;
};

// ===================================================================================================================
// Helper functions

// Square root
// o Overloaded versions of the sqrt function
// o The square root of a complex number is expensive.  However, the diagonal values of a Cholesky decomposition are
// always
//   real so we don't need a full complex square root.
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(T_IN a, T_OUT& b) {
Function_cholesky_sqrt_op_real:;
    const T_IN ZERO = 0;
    if (a < ZERO) {
        b = ZERO;
        return (1);
    }
    b = x_sqrt(a);
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(hls::x_complex<T_IN> din, hls::x_complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_complex:;
    const T_IN ZERO = 0;
    T_IN a = din.real();
    dout.imag(ZERO);

    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }

    dout.real(x_sqrt(a));
    return (0);
}
template <typename T_IN, typename T_OUT>
int cholesky_sqrt_op(std::complex<T_IN> din, std::complex<T_OUT>& dout) {
Function_cholesky_sqrt_op_complex:;
    const T_IN ZERO = 0;
    T_IN a = din.real();
    dout.imag(ZERO);

    if (a < ZERO) {
        dout.real(ZERO);
        return (1);
    }

    dout.real(x_sqrt(a));
    return (0);
}

// Reciprocal square root.
template <typename InputType, typename OutputType>
void cholesky_rsqrt(InputType x, OutputType& res) {
Function_cholesky_rsqrt_default:;
    res = x_rsqrt(x);
}
template <int W1, int I1, ap_q_mode Q1, ap_o_mode O1, int N1, int W2, int I2, ap_q_mode Q2, ap_o_mode O2, int N2>
void cholesky_rsqrt(ap_fixed<W1, I1, Q1, O1, N1> x, ap_fixed<W2, I2, Q2, O2, N2>& res) {
Function_cholesky_rsqrt_fixed:;
    ap_fixed<W2, I2, Q2, O2, N2> one = 1;
    ap_fixed<W1, I1, Q1, O1, N1> sqrt_res;
    ap_fixed<W2, I2, Q2, O2, N2> sqrt_res_cast;
    sqrt_res = x_sqrt(x);
    sqrt_res_cast = sqrt_res;
    res = one / sqrt_res_cast;
}

// Local multiplier to handle a complex case currently not supported by the hls::x_complex class
// - Complex multiplied by a real of a different type
// - Required for complex fixed point implementations
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(AType A, BType B, CType& C) {
Function_cholesky_prod_sum_mult_real:;
    C = A * B;
}
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(hls::x_complex<AType> A, BType B, hls::x_complex<CType>& C) {
Function_cholesky_prod_sum_mult_complex:;
    C.real(A.real() * B);
    C.imag(A.imag() * B);
}
template <typename AType, typename BType, typename CType>
void cholesky_prod_sum_mult(std::complex<AType> A, BType B, std::complex<CType>& C) {
Function_cholesky_prod_sum_mult_complex:;
    C.real(A.real() * B);
    C.imag(A.imag() * B);
}

namespace details {

template <typename T>
inline void compute_small_diag_scalars(const T& diag_real, T& diag_scalar, T& inv_diag) {
#pragma HLS INLINE
    inv_diag = x_rsqrt(diag_real);
    diag_scalar = diag_real * inv_diag;
}

template <int W, int I, ap_q_mode Q, ap_o_mode O, int N>
inline void compute_small_diag_scalars(const ap_fixed<W, I, Q, O, N>& diag_real,
                                       ap_fixed<W, I, Q, O, N>& diag_scalar,
                                       ap_fixed<W, I, Q, O, N>& inv_diag) {
#pragma HLS INLINE
    float diag_real_f = static_cast<float>(diag_real);
    float inv_diag_f = fast_inv_sqrt_float(diag_real_f);
    float diag_val_f = diag_real_f * inv_diag_f;
    inv_diag = inv_diag_f;
    diag_scalar = diag_val_f;
}

template <typename CholeskyTraits>
inline bool finalize_small_diag(typename CholeskyTraits::ACCUM_T diag_term,
                                typename CholeskyTraits::L_OUTPUT_T& diag_store,
                                typename CholeskyTraits::RECIP_DIAG_T& recip_val) {
#pragma HLS INLINE
    typename CholeskyTraits::DIAG_T diag_cast = diag_term;
    typename CholeskyTraits::RECIP_DIAG_T diag_real = hls::x_real(diag_cast);
    typename CholeskyTraits::RECIP_DIAG_T zero = 0;
    if (diag_real < 0) {
#ifndef __SYNTHESIS__
        printf("ERROR: Trying to find the square root of a negative number\n");
#endif
        recip_val = 0;
        set_real_zero_imag(diag_store, zero);
        return true;
    }

    typename CholeskyTraits::RECIP_DIAG_T diag_scalar;
    typename CholeskyTraits::RECIP_DIAG_T inv_diag;
    compute_small_diag_scalars(diag_real, diag_scalar, inv_diag);

    set_real_zero_imag(diag_store, diag_scalar);
    recip_val = inv_diag;
    return false;
}

template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType>
inline typename CholeskyTraits::ACCUM_T fetch_small_core_value(const InputType A[RowsColsA][RowsColsA],
                                                               int row,
                                                               int col) {
#pragma HLS INLINE
    if (LowerTriangularL) {
        return typename CholeskyTraits::ACCUM_T(A[row][col]);
    } else {
        return typename CholeskyTraits::ACCUM_T(hls::x_conj(A[col][row]));
    }
}

} // namespace details

// ===================================================================================================================
// Specialized fully-unrolled implementation for very small matrices (RowsColsA <= 4)
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyAlt(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
#pragma HLS INLINE off
#pragma HLS BIND_OP op=mul impl=DSP
    const int N = RowsColsA;

    typename CholeskyTraits::L_OUTPUT_T L_internal[RowsColsA][RowsColsA];
    typename CholeskyTraits::RECIP_DIAG_T recip_diag[RowsColsA];
#pragma HLS ARRAY_PARTITION variable = L_internal complete dim = 0
#pragma HLS ARRAY_PARTITION variable = recip_diag complete dim = 1
#pragma HLS ARRAY_PARTITION variable = A complete dim = 0
#pragma HLS ARRAY_PARTITION variable = L complete dim = 0
#pragma HLS bind_storage variable = L_internal type = ram_2p impl = register
#pragma HLS bind_storage variable = A type = ram_2p impl = register
#pragma HLS LATENCY max=180

    // Initialize outputs and internal buffers.
    for (int r = 0; r < N; ++r) {
#pragma HLS UNROLL
        for (int c = 0; c < N; ++c) {
#pragma HLS UNROLL
            L_internal[r][c] = 0;
            L[r][c] = 0;
        }
    }

    if (N > 0) {
        typename CholeskyTraits::ACCUM_T diag_term = A[0][0];
        typename CholeskyTraits::L_OUTPUT_T diag_store;
        if (details::finalize_small_diag<CholeskyTraits>(diag_term, diag_store, recip_diag[0])) {
            return 1;
        }
        L_internal[0][0] = diag_store;

        for (int i = 1; i < N; ++i) {
#pragma HLS UNROLL
            typename CholeskyTraits::ACCUM_T val =
                details::fetch_small_core_value<LowerTriangularL, RowsColsA, CholeskyTraits>(A, i, 0);
            typename CholeskyTraits::OFF_DIAG_T off_value;
            cholesky_prod_sum_mult(val, recip_diag[0], off_value);
            L_internal[i][0] = off_value;
        }
    }

    if (N > 1) {
        typename CholeskyTraits::ACCUM_T diag_term = A[1][1];
        typename CholeskyTraits::L_OUTPUT_T l10 = L_internal[1][0];
        // Use magnitude squared (real domain) for efficiency
        auto msq0 = details::magnitude_squared(l10);
        typename CholeskyTraits::ACCUM_T contrib;
        details::set_real_zero_imag(contrib, msq0);
        diag_term -= contrib;

        typename CholeskyTraits::L_OUTPUT_T diag_store;
        if (details::finalize_small_diag<CholeskyTraits>(diag_term, diag_store, recip_diag[1])) {
            return 1;
        }
        L_internal[1][1] = diag_store;

        for (int i = 2; i < N; ++i) {
#pragma HLS UNROLL
            typename CholeskyTraits::ACCUM_T val =
                details::fetch_small_core_value<LowerTriangularL, RowsColsA, CholeskyTraits>(A, i, 1);
            typename CholeskyTraits::L_OUTPUT_T lik = L_internal[i][0];
            typename CholeskyTraits::L_OUTPUT_T ljk = L_internal[1][0];
            typename CholeskyTraits::ACCUM_T prod = lik * hls::x_conj(ljk);
            val -= prod;

            typename CholeskyTraits::OFF_DIAG_T off_value;
            cholesky_prod_sum_mult(val, recip_diag[1], off_value);
            L_internal[i][1] = off_value;
        }
    }

    if (N > 2) {
        typename CholeskyTraits::ACCUM_T diag_term = A[2][2];
        typename CholeskyTraits::L_OUTPUT_T l20 = L_internal[2][0];
        typename CholeskyTraits::L_OUTPUT_T l21 = L_internal[2][1];
        // Merge two subtractions: sum in real domain first, then one complex subtraction
        auto msq0 = details::magnitude_squared(l20);
        auto msq1 = details::magnitude_squared(l21);
        typename CholeskyTraits::ACCUM_T contrib;
        details::set_real_zero_imag(contrib, msq0 + msq1);
        diag_term -= contrib;

        typename CholeskyTraits::L_OUTPUT_T diag_store;
        if (details::finalize_small_diag<CholeskyTraits>(diag_term, diag_store, recip_diag[2])) {
            return 1;
        }
        L_internal[2][2] = diag_store;

        for (int i = 3; i < N; ++i) {
#pragma HLS UNROLL
            typename CholeskyTraits::ACCUM_T val =
                details::fetch_small_core_value<LowerTriangularL, RowsColsA, CholeskyTraits>(A, i, 2);
            typename CholeskyTraits::L_OUTPUT_T lik0 = L_internal[i][0];
            typename CholeskyTraits::L_OUTPUT_T ljk0 = L_internal[2][0];
            typename CholeskyTraits::ACCUM_T prod0 = lik0 * hls::x_conj(ljk0);
            val -= prod0;

            typename CholeskyTraits::L_OUTPUT_T lik1 = L_internal[i][1];
            typename CholeskyTraits::L_OUTPUT_T ljk1 = L_internal[2][1];
            typename CholeskyTraits::ACCUM_T prod1 = lik1 * hls::x_conj(ljk1);
            val -= prod1;

            typename CholeskyTraits::OFF_DIAG_T off_value;
            cholesky_prod_sum_mult(val, recip_diag[2], off_value);
            L_internal[i][2] = off_value;
        }
    }

    if (N > 3) {
        typename CholeskyTraits::ACCUM_T diag_term = A[3][3];
        typename CholeskyTraits::L_OUTPUT_T l30 = L_internal[3][0];
        typename CholeskyTraits::L_OUTPUT_T l31 = L_internal[3][1];
        typename CholeskyTraits::L_OUTPUT_T l32 = L_internal[3][2];

        // Merge three subtractions: sum in real domain first, then one complex subtraction
        auto msq0 = details::magnitude_squared(l30);
        auto msq1 = details::magnitude_squared(l31);
        auto msq2 = details::magnitude_squared(l32);
        typename CholeskyTraits::ACCUM_T contrib;
        details::set_real_zero_imag(contrib, msq0 + msq1 + msq2);
        diag_term -= contrib;

        typename CholeskyTraits::L_OUTPUT_T diag_store;
        if (details::finalize_small_diag<CholeskyTraits>(diag_term, diag_store, recip_diag[3])) {
            return 1;
        }
        L_internal[3][3] = diag_store;
    }

    for (int r = 0; r < N; ++r) {
#pragma HLS UNROLL
        for (int c = 0; c < N; ++c) {
#pragma HLS UNROLL
            if (LowerTriangularL) {
                if (c <= r) {
                    L[r][c] = L_internal[r][c];
                } else {
                    L[r][c] = 0;
                }
            } else {
                if (r <= c) {
                    L[r][c] = details::conj_copy(L_internal[c][r]);
                } else {
                    L[r][c] = 0;
                }
            }
        }
    }
    return 0;
}

// ===================================================================================================================
// choleskyBasic
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyBasic(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    int return_code = 0;

    // Use the traits struct to specify the correct type for the intermediate variables. This is really only needed for
    // fixed point.
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T sum[RowsColsA];
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;    // A with the same dimensions as sum.
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum; // prod with the same dimensions as sum.

    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T new_L_diag;         // sqrt(A_minus_sum)
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag; // sum/L
    typename CholeskyTraits::OFF_DIAG_T L_cast_to_new_L_off_diag;

    typename CholeskyTraits::L_OUTPUT_T new_L;
    OutputType retrieved_L;
    // Internal memory used to aviod read access from function output argument L.
    // NOTE: The internal matrix only needs to be triangular but optimization using a 1-D array it will require addition
    // logic to generate the indexes. Refer to the choleskyAlt function.
    OutputType L_internal[RowsColsA][RowsColsA];

col_loop:
    for (int j = 0; j < RowsColsA; j++) {
        sum[j] = 0;

    // Calculate the diagonal value for this column
    diag_loop:
        for (int k = 0; k < RowsColsA; k++) {
            if (k <= (j - 1)) {
                if (LowerTriangularL == true) {
                    retrieved_L = L_internal[j][k];
                } else {
                    retrieved_L = L_internal[k][j];
                }
                sum[j] = hls::x_conj(retrieved_L) * retrieved_L;
            }
        }
        A_cast_to_sum = A[j][j];

        A_minus_sum = A_cast_to_sum - sum[j];

        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }

        // Round to target format using method specifed by traits defined types.
        new_L = new_L_diag;

        if (LowerTriangularL == true) {
            L_internal[j][j] = new_L;
            L[j][j] = new_L;
        } else {
            L_internal[j][j] = hls::x_conj(new_L);
            L[j][j] = hls::x_conj(new_L);
        }

    // Calculate the off diagonal values for this column
    off_diag_loop:
        for (int i = 0; i < RowsColsA; i++) {
            if (i > j) {
                if (LowerTriangularL == true) {
                    sum[j] = A[i][j];
                } else {
                    sum[j] = hls::x_conj(A[j][i]);
                }

            sum_loop:
                for (int k = 0; k < RowsColsA; k++) {
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
                    if (k <= (j - 1)) {
                        if (LowerTriangularL == true) {
                            prod = -L_internal[i][k] * hls::x_conj(L_internal[j][k]);
                        } else {
                            prod = -hls::x_conj(L_internal[k][i]) * (L_internal[k][j]);
                        }

                        prod_cast_to_sum = prod;
                        sum[j] += prod_cast_to_sum;
                    }
                }

                new_L_off_diag = sum[j];

                L_cast_to_new_L_off_diag = L_internal[j][j];

                // Diagonal is always real, avoid complex division
                new_L_off_diag = new_L_off_diag / hls::x_real(L_cast_to_new_L_off_diag);

                // Round to target format using method specifed by traits defined types.
                new_L = new_L_off_diag;

                if (LowerTriangularL == true) {
                    L[i][j] = new_L;
                    L_internal[i][j] = new_L;
                } else {
                    L[j][i] = hls::x_conj(new_L);
                    L_internal[j][i] = hls::x_conj(new_L);
                }
            } else if (i < j) {
                if (LowerTriangularL == true) {
                    L[i][j] = 0;
                } else {
                    L[j][i] = 0;
                }
            }
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyAlt: Alternative architecture with improved latency at the expense of higher resource
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskySmall(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
#pragma HLS BIND_OP op=mul impl=DSP
#pragma HLS BIND_OP op=add impl=Fabric
#pragma HLS ALLOCATION operation instances=mul limit=48
#pragma HLS ALLOCATION operation instances=add limit=96
    int return_code = 0;

    // Optimize internal memories
    // - For complex data types the diagonal will be real only, plus for fixed point it must be stored to a
    //   higher precision.
    // - Requires additional logic to generate the memory indexes
    // - For smaller matrix sizes there maybe be an increase in memory usage
    OutputType L_internal[(RowsColsA * RowsColsA - RowsColsA) / 2];
    typename CholeskyTraits::RECIP_DIAG_T diag_internal[RowsColsA];

    typename CholeskyTraits::ACCUM_T square_sum;
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;
    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T A_minus_sum_cast_diag;
    typename CholeskyTraits::DIAG_T new_L_diag;
    typename CholeskyTraits::RECIP_DIAG_T new_L_diag_recip;
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum;
    typename CholeskyTraits::ACCUM_T product_sum;
    typename CholeskyTraits::OFF_DIAG_T prod_cast_to_off_diag;
    typename CholeskyTraits::RECIP_DIAG_T L_diag_recip;
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag;
    typename CholeskyTraits::L_OUTPUT_T new_L;
    typename CholeskyTraits::L_OUTPUT_T new_L_recip;

row_loop:
    for (int i = 0; i < RowsColsA; i++) {
        // Index generation for optimized/packed L_internal memory
        int i_sub1 = i - 1;
        int i_off = ((i_sub1 * i_sub1 - i_sub1) / 2) + i_sub1;

        // Off diagonal calculation
        square_sum = 0;
    col_loop:
        for (int j = 0; j < i; j++) {
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
            // Index generation
            int j_sub1 = j - 1;
            int j_off = ((j_sub1 * j_sub1 - j_sub1) / 2) + j_sub1;
            // Prime the off-diagonal sum with target elements A value.
            if (LowerTriangularL == true) {
                product_sum = A[i][j];
            } else {
                product_sum = hls::x_conj(A[j][i]);
            }
        sum_loop:
            for (int k = 0; k < j; k++) {
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
                prod = -L_internal[i_off + k] * hls::x_conj(L_internal[j_off + k]);
                prod_cast_to_sum = prod;
                product_sum += prod_cast_to_sum;
            }
            prod_cast_to_off_diag = product_sum;
            // Fetch diagonal value
            L_diag_recip = diag_internal[j];
            // Diagonal is stored in its reciprocal form so only need to multiply the product sum
            cholesky_prod_sum_mult(prod_cast_to_off_diag, L_diag_recip, new_L_off_diag);
            // Round to target format using method specifed by traits defined types.
            new_L = new_L_off_diag;
            // Build sum for use in diagonal calculation for this row.
            // Use magnitude squared for efficiency (real domain)
            auto new_L_msq = details::magnitude_squared(new_L);
            typename CholeskyTraits::ACCUM_T new_L_msq_complex;
            details::set_real_zero_imag(new_L_msq_complex, new_L_msq);
            square_sum += new_L_msq_complex;
            // Store result
            L_internal[i_off + j] = new_L;
            if (LowerTriangularL == true) {
                L[i][j] = new_L; // store in lower triangle
                L[j][i] = 0;     // Zero upper
            } else {
                L[j][i] = hls::x_conj(new_L); // store in upper triangle
                L[i][j] = 0;                  // Zero lower
            }
        }

        // Diagonal calculation
        A_cast_to_sum = A[i][i];
        A_minus_sum = A_cast_to_sum - square_sum;
        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }
        // Round to target format using method specifed by traits defined types.
        new_L = new_L_diag;
        // Generate the reciprocal of the diagonal for internal use to aviod the latency of a divide in every
        // off-diagonal calculation
        A_minus_sum_cast_diag = A_minus_sum;
        cholesky_rsqrt(hls::x_real(A_minus_sum_cast_diag), new_L_diag_recip);
        // Store diagonal value
        diag_internal[i] = new_L_diag_recip;
        if (LowerTriangularL == true) {
            L[i][i] = new_L;
        } else {
            L[i][i] = hls::x_conj(new_L);
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyAlt2: Further improved latency architecture requiring higher resource
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyAlt2(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
#pragma HLS BIND_OP op=mul impl=DSP
#pragma HLS BIND_OP op=add impl=Fabric
#pragma HLS ALLOCATION operation instances=mul limit=64
#pragma HLS ALLOCATION operation instances=add limit=128
    int return_code = 0;

    // To avoid array index calculations every iteration this architecture uses a simple 2D array rather than a
    // optimized/packed triangular matrix.
    OutputType L_internal[RowsColsA][RowsColsA];
    OutputType prod_column_top;
    typename CholeskyTraits::ACCUM_T square_sum_array[RowsColsA];
    typename CholeskyTraits::ACCUM_T A_cast_to_sum;
    typename CholeskyTraits::ADD_T A_minus_sum;
    typename CholeskyTraits::DIAG_T A_minus_sum_cast_diag;
    typename CholeskyTraits::DIAG_T new_L_diag;
    typename CholeskyTraits::RECIP_DIAG_T new_L_diag_recip;
    typename CholeskyTraits::PROD_T prod;
    typename CholeskyTraits::ACCUM_T prod_cast_to_sum;
    typename CholeskyTraits::ACCUM_T product_sum;
    typename CholeskyTraits::ACCUM_T product_sum_array[RowsColsA];
    typename CholeskyTraits::OFF_DIAG_T prod_cast_to_off_diag;
    typename CholeskyTraits::OFF_DIAG_T new_L_off_diag;
    typename CholeskyTraits::L_OUTPUT_T new_L;

#pragma HLS ARRAY_PARTITION variable = A cyclic dim = CholeskyTraits::UNROLL_DIM factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = L cyclic dim = CholeskyTraits::UNROLL_DIM factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = L_internal cyclic dim = CholeskyTraits::UNROLL_DIM factor = \
    CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = square_sum_array cyclic dim = 1 factor = CholeskyTraits::UNROLL_FACTOR
#pragma HLS ARRAY_PARTITION variable = product_sum_array cyclic dim = 1 factor = CholeskyTraits::UNROLL_FACTOR

col_loop:
    for (int j = 0; j < RowsColsA; j++) {
        // Diagonal calculation
        A_cast_to_sum = A[j][j];
        if (j == 0) {
            A_minus_sum = A_cast_to_sum;
        } else {
            A_minus_sum = A_cast_to_sum - square_sum_array[j];
        }
        if (cholesky_sqrt_op(A_minus_sum, new_L_diag)) {
#ifndef __SYNTHESIS__
            printf("ERROR: Trying to find the square root of a negative number\n");
#endif
            return_code = 1;
        }
        // Round to target format using method specifed by traits defined types.
        new_L = new_L_diag;
        // Generate the reciprocal of the diagonal for internal use to aviod the latency of a divide in every
        // off-diagonal calculation
        A_minus_sum_cast_diag = A_minus_sum;
        cholesky_rsqrt(hls::x_real(A_minus_sum_cast_diag), new_L_diag_recip);
        // Store diagonal value
        if (LowerTriangularL == true) {
            L[j][j] = new_L;
        } else {
            L[j][j] = hls::x_conj(new_L);
        }

    sum_loop:
        for (int k = 0; k <= j; k++) {
// Define average trip count for reporting, loop reduces in length for every iteration of col_loop
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
            // Same value used in all calcs
            // o Implement -1* here
            prod_column_top = -hls::x_conj(L_internal[j][k]);

        // NOTE: Using a fixed loop length combined with a "if" to implement reducing loop length
        // o Ensures the inner loop can achieve the maximum II (1)
        // o May introduce a small overhead resolving the "if" statement but HLS struggled to schedule when the variable
        //   loop bound expression was used.
        // o Will report inaccurate trip count as it will reduce by one with the col_loop
        // o Variable loop bound code: row_loop: for(int i = j+1; i < RowsColsA; i++) {
        row_loop:
            for (int i = 0; i < RowsColsA; i++) {
// IMPORTANT: row_loop must not merge with sum_loop as the merged loop becomes variable length and HLS will struggle
// with scheduling
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = CholeskyTraits::INNER_II
#pragma HLS UNROLL FACTOR = CholeskyTraits::UNROLL_FACTOR

                if (i > j) {
                    prod = L_internal[i][k] * prod_column_top;
                    prod_cast_to_sum = prod;

                    if (k == 0) {
                        // Prime first sum
                        if (LowerTriangularL == true) {
                            A_cast_to_sum = A[i][j];
                        } else {
                            A_cast_to_sum = hls::x_conj(A[j][i]);
                        }
                        product_sum = A_cast_to_sum;
                    } else {
                        product_sum = product_sum_array[i];
                    }

                    if (k < j) {
                        // Accumulate row sum of columns
                        product_sum_array[i] = product_sum + prod_cast_to_sum;
                    } else {
                        // Final calculation for off diagonal value
                        prod_cast_to_off_diag = product_sum;
                        // Diagonal is stored in its reciprocal form so only need to multiply the product sum
                        cholesky_prod_sum_mult(prod_cast_to_off_diag, new_L_diag_recip, new_L_off_diag);
                        // Round to target format using method specifed by traits defined types.
                        new_L = new_L_off_diag;
                        // Build sum for use in diagonal calculation for this row.
                        // Use magnitude squared for efficiency (real domain)
                        auto new_L_msq = details::magnitude_squared(new_L);
                        details::set_real_zero_imag(square_sum_array[j], new_L_msq);
                        // Store result
                        L_internal[i][j] = new_L;
                        // NOTE: Use the upper/lower triangle zeroing in the subsequent loop so the double memory access
                        // does not
                        // become a bottleneck
                        // o Results in a further increase of DSP resources due to the higher II of this loop.
                        // o Retaining the zeroing operation here can give this a loop a max II of 2 and HLS will
                        // resource share.
                        if (LowerTriangularL == true) {
                            L[i][j] = new_L;                                   // Store in lower triangle
                            if (!CholeskyTraits::ARCH2_ZERO_LOOP) L[j][i] = 0; // Zero upper
                        } else {
                            L[j][i] = hls::x_conj(new_L);                      // Store in upper triangle
                            if (!CholeskyTraits::ARCH2_ZERO_LOOP) L[i][j] = 0; // Zero lower
                        }
                    }
                }
            }
        }
    }
    // Zero upper/lower triangle
    // o Use separate loop to ensure main calcuation can achieve an II of 1
    // o As noted above this may increase the DSP resources.
    // o Required when unrolling the inner loop due to array dimension access
    if (CholeskyTraits::ARCH2_ZERO_LOOP) {
    zero_rows_loop:
        for (int i = 0; i < RowsColsA - 1; i++) {
        zero_cols_loop:
            for (int j = i + 1; j < RowsColsA; j++) {
// Define average trip count for reporting, loop reduces in length for every iteration of zero_rows_loop
#pragma HLS loop_tripcount max = 1 + RowsColsA / 2
#pragma HLS PIPELINE
                if (LowerTriangularL == true) {
                    L[i][j] = 0; // Zero upper
                } else {
                    L[j][i] = 0; // Zero lower
                }
            }
        }
    }
    return (return_code);
}

// ===================================================================================================================
// choleskyTop: Top level function that selects implementation architecture and internal types based on the
// traits class provided via the CholeskyTraits template parameter.
// o Call this function directly if you wish to override the default architecture choice or internal types
template <bool LowerTriangularL, int RowsColsA, typename CholeskyTraits, class InputType, class OutputType>
int choleskyTop(const InputType A[RowsColsA][RowsColsA], OutputType L[RowsColsA][RowsColsA]) {
    if (details::cholesky_opt_config<RowsColsA>::USE_SMALL_CORE) {
        return choleskyAlt<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
    }
    switch (CholeskyTraits::ARCH) {
        case 0:
            return choleskyBasic<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        case 1:
            return choleskyAlt<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        case 2:
            return choleskyAlt2<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
        default:
            return choleskyBasic<LowerTriangularL, RowsColsA, CholeskyTraits, InputType, OutputType>(A, L);
    }
}

/**
* @brief cholesky
*
* @tparam LowerTriangularL   When false generates the result in the upper triangle
* @tparam RowsColsA          Defines the matrix dimensions
* @tparam InputType          Input data type
* @tparam OutputType         Output data type
* @tparam TRAITS             choleskyTraits class
*
* @param matrixAStrm         Stream of Hermitian/symmetric positive definite input matrix
* @param matrixLStrm         Stream of Lower or upper triangular output matrix
*
* @return                    An integer type. 0=Success. 1=Failure. The function attempted to find the square root of
* a negative number i.e. the input matrix A was not Hermitian/symmetric positive definite.
*/
template <bool LowerTriangularL,
          int RowsColsA,
          class InputType,
          class OutputType,
          typename TRAITS = choleskyTraits<LowerTriangularL, RowsColsA, InputType, OutputType> >
int cholesky(hls::stream<InputType>& matrixAStrm, hls::stream<OutputType>& matrixLStrm) {
    InputType A[RowsColsA][RowsColsA];
    OutputType L[RowsColsA][RowsColsA];
#pragma HLS ARRAY_PARTITION variable = A complete dim = 0
#pragma HLS ARRAY_PARTITION variable = L complete dim = 0

    const int total_elems = RowsColsA * RowsColsA;

read_matrix_loop:
    for (int idx = 0; idx < total_elems; ++idx) {
#pragma HLS PIPELINE II = 1
        int r = idx / RowsColsA;
        int c = idx % RowsColsA;
        matrixAStrm.read(A[r][c]);
    }

    int ret = 0;
    ret = choleskyTop<LowerTriangularL, RowsColsA, TRAITS, InputType, OutputType>(A, L);

write_matrix_loop:
    for (int idx = 0; idx < total_elems; ++idx) {
#pragma HLS PIPELINE II = 1
        int r = idx / RowsColsA;
        int c = idx % RowsColsA;
        matrixLStrm.write(L[r][c]);
    }
    return ret;
}

} // end namespace solver
} // end namespace xf
#endif
