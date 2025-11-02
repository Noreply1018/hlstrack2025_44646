/*
 * Copyright 2019 Xilinx, Inc.
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

#ifndef _XF_SECURITY_SHA224_256_HPP_
#define _XF_SECURITY_SHA224_256_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "xf_security/types.hpp"
#include "xf_security/utils.hpp"

// For debug
#ifndef __SYNTHESIS__
#include <cstdio>
#endif
#ifndef _DEBUG
#define _DEBUG (0)
#endif
#define _XF_SECURITY_VOID_CAST static_cast<void>
// XXX toggle here to debug this file
#define _XF_SECURITY_PRINT(msg...) \
    do {                           \
        if (_DEBUG) printf(msg);   \
    } while (0)

// Inline rotation and shift operations for better optimization
static inline uint32_t ROTR_func(unsigned int n, uint32_t x) {
#pragma HLS inline
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t SHR_func(unsigned int n, uint32_t x) {
#pragma HLS inline
    return x >> n;
}

// Optimized CH: reduce from 3 to 2 logic levels
static inline uint32_t CH_func(uint32_t x, uint32_t y, uint32_t z) {
#pragma HLS inline
    return (x & (y ^ z)) ^ z;
}

// Optimized MAJ: reduce XOR chain
static inline uint32_t MAJ_func(uint32_t x, uint32_t y, uint32_t z) {
#pragma HLS inline
    return ((x | y) & z) | (x & y);
}

// BSIG/SSIG as inline functions with reduced logic depth
static inline uint32_t BSIG0_func(uint32_t x) {
#pragma HLS inline
    return ROTR_func(2, x) ^ (ROTR_func(13, x) ^ ROTR_func(22, x));
}

static inline uint32_t BSIG1_func(uint32_t x) {
#pragma HLS inline
    return ROTR_func(6, x) ^ (ROTR_func(11, x) ^ ROTR_func(25, x));
}

static inline uint32_t SSIG0_func(uint32_t x) {
#pragma HLS inline
    return ROTR_func(7, x) ^ (ROTR_func(18, x) ^ SHR_func(3, x));
}

static inline uint32_t SSIG1_func(uint32_t x) {
#pragma HLS inline
    return ROTR_func(17, x) ^ (ROTR_func(19, x) ^ SHR_func(10, x));
}

// Keep old macros for backward compatibility
#define ROTR(n, x) ROTR_func(n, x)
#define ROTL(n, x) ((x << n) | (x >> (32 - n)))
#define SHR(n, x) SHR_func(n, x)
#define CH(x, y, z) CH_func(x, y, z)
#define MAJ(x, y, z) MAJ_func(x, y, z)
#define BSIG0(x) BSIG0_func(x)
#define BSIG1(x) BSIG1_func(x)
#define SSIG0(x) SSIG0_func(x)
#define SSIG1(x) SSIG1_func(x)

namespace xf {
namespace security {
namespace internal {

/// Processing block
struct SHA256Block {
    uint32_t M[16];
};

/// @brief Static config for SHA224 and SHA256.
template <bool do_sha224>
struct sha256_digest_config;

template <>
struct sha256_digest_config<true> {
    static const short numH = 7;
};

template <>
struct sha256_digest_config<false> {
    static const short numH = 8;
};

// Precomputed SHA-256 round constants.
static const uint32_t SHA256_K[64] = {
    0x428a2f98UL, 0x71374491UL, 0xb5c0fbcfUL, 0xe9b5dba5UL, 0x3956c25bUL, 0x59f111f1UL, 0x923f82a4UL, 0xab1c5ed5UL,
    0xd807aa98UL, 0x12835b01UL, 0x243185beUL, 0x550c7dc3UL, 0x72be5d74UL, 0x80deb1feUL, 0x9bdc06a7UL, 0xc19bf174UL,
    0xe49b69c1UL, 0xefbe4786UL, 0x0fc19dc6UL, 0x240ca1ccUL, 0x2de92c6fUL, 0x4a7484aaUL, 0x5cb0a9dcUL, 0x76f988daUL,
    0x983e5152UL, 0xa831c66dUL, 0xb00327c8UL, 0xbf597fc7UL, 0xc6e00bf3UL, 0xd5a79147UL, 0x06ca6351UL, 0x14292967UL,
    0x27b70a85UL, 0x2e1b2138UL, 0x4d2c6dfcUL, 0x53380d13UL, 0x650a7354UL, 0x766a0abbUL, 0x81c2c92eUL, 0x92722c85UL,
    0xa2bfe8a1UL, 0xa81a664bUL, 0xc24b8b70UL, 0xc76c51a3UL, 0xd192e819UL, 0xd6990624UL, 0xf40e3585UL, 0x106aa070UL,
    0x19a4c116UL, 0x1e376c08UL, 0x2748774cUL, 0x34b0bcb5UL, 0x391c0cb3UL, 0x4ed8aa4aUL, 0x5b9cca4fUL, 0x682e6ff3UL,
    0x748f82eeUL, 0x78a5636fUL, 0x84c87814UL, 0x8cc70208UL, 0x90befffaUL, 0xa4506cebUL, 0xbef9a3f7UL, 0xc67178f2UL};

static const uint32_t SHA256_IV256[8] = {
    0x6a09e667UL, 0xbb67ae85UL, 0x3c6ef372UL, 0xa54ff53aUL, 0x510e527fUL, 0x9b05688cUL, 0x1f83d9abUL, 0x5be0cd19UL};

static const uint32_t SHA256_IV224[8] = {
    0xc1059ed8UL, 0x367cd507UL, 0x3070dd17UL, 0xf70e5939UL, 0xffc00b31UL, 0x68581511UL, 0x64f98fa7UL, 0xbefa4fa4UL};

inline void unpackState(ap_uint<256> packed, uint32_t state[8]) {
#pragma HLS inline
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        state[i] = packed.range(32 * (i + 1) - 1, 32 * i);
    }
}

inline ap_uint<256> packState(const uint32_t state[8]) {
#pragma HLS inline
    ap_uint<256> packed = 0;
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        packed.range(32 * (i + 1) - 1, 32 * i) = state[i];
    }
    return packed;
}

inline void blockToWords(ap_uint<512> block, uint32_t words[16]) {
#pragma HLS inline
    for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
        int hi = 512 - 1 - 32 * i;
        int lo = hi - 31;
        ap_uint<32> raw = block.range(hi, lo);
        uint32_t tmp = (uint32_t)raw;
        uint32_t swapped = ((tmp & 0x000000ffUL) << 24) | ((tmp & 0x0000ff00UL) << 8) |
                           ((tmp & 0x00ff0000UL) >> 8) | ((tmp & 0xff000000UL) >> 24);
        words[i] = swapped;
    }
}

inline void sha256CompressBlock(const uint32_t words[16], uint32_t state[8]) {
#pragma HLS inline
#pragma HLS array_partition variable = SHA256_K complete
#pragma HLS bind_storage variable = SHA256_K type = rom_1p impl = lutram

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    uint32_t W16[16];
#pragma HLS array_partition variable = W16 complete
    for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
        W16[i] = words[i];
    }

LOOP_SHA256_COMPRESS:
    for (int t = 0; t < 64; ++t) {
#pragma HLS pipeline II = 1 rewind
        uint32_t Wt;
        if (t < 16) {
            Wt = W16[t];
        } else {
            uint32_t w_tm2 = W16[(t - 2) & 0xF];
            uint32_t w_tm7 = W16[(t - 7) & 0xF];
            uint32_t w_tm15 = W16[(t - 15) & 0xF];
            uint32_t w_tm16 = W16[(t - 16) & 0xF];
            ap_uint<33> sum_a = (ap_uint<33>)SSIG0(w_tm15) + SSIG1(w_tm2);
            ap_uint<33> sum_b = (ap_uint<33>)w_tm7 + w_tm16;
            ap_uint<34> sum_ab = (ap_uint<34>)sum_b + (ap_uint<34>)sum_a;
            Wt = (uint32_t)sum_ab;
            W16[t & 0xF] = Wt;
        }

        // Compute T1 and T2 with optimized datapath
        uint32_t Kt_local = SHA256_K[t];
        
        // T1 path
        ap_uint<33> t1_h_bsig1 = (ap_uint<33>)h + BSIG1(e);
        ap_uint<33> t1_ch_wt = (ap_uint<33>)CH(e, f, g) + Wt;
        ap_uint<34> t1_mid = (ap_uint<34>)t1_h_bsig1 + (ap_uint<34>)t1_ch_wt;
        ap_uint<35> t1_final = t1_mid + (ap_uint<35>)Kt_local;
        uint32_t T1 = (uint32_t)t1_final;
        
        // T2 path (parallel to T1)
        ap_uint<33> t2_final = (ap_uint<33>)BSIG0(a) + MAJ(a, b, c);
        uint32_t T2 = (uint32_t)t2_final;

        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    state[0] = state[0] + a;
    state[1] = state[1] + b;
    state[2] = state[2] + c;
    state[3] = state[3] + d;
    state[4] = state[4] + e;
    state[5] = state[5] + f;
    state[6] = state[6] + g;
    state[7] = state[7] + h;
}

inline ap_uint<256> sha256PrecomputePad(ap_uint<512> pad_block, const uint32_t init_state[8]) {
#pragma HLS inline
    uint32_t words[16];
#pragma HLS array_partition variable = words complete
    blockToWords(pad_block, words);
    uint32_t state[8];
#pragma HLS array_partition variable = state complete
    for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
        state[i] = init_state[i];
    }
    sha256CompressBlock(words, state);
    return packState(state);
}

/// @brief Generate 512bit processing blocks for SHA224/SHA256 (pipeline)
/// with const width.
/// The performance goal of this function is to yield a 512b block per cycle.
/// @param msg_strm the message being hashed.
/// @param len_strm the message length in byte.
/// @param end_len_strm that flag to signal end of input.
/// @param blk_strm the 512-bit hash block.
/// @param nblk_strm the number of hash block for this message.
/// @param end_nblk_strm end flag for number of hash block.
inline void preProcessing(hls::stream<ap_uint<32> >& msg_strm,
                          hls::stream<ap_uint<64> >& len_strm,
                          hls::stream<bool>& end_len_strm,
                          hls::stream<SHA256Block>& blk_strm,
                          hls::stream<uint64_t>& nblk_strm,
                          hls::stream<bool>& end_nblk_strm,
                          uint64_t pre_len = 0) {
LOOP_SHA256_GENENERATE_MAIN:
    for (bool end_flag = end_len_strm.read(); !end_flag; end_flag = end_len_strm.read()) {
        /// message length in byte.
        uint64_t len = len_strm.read();
        /// message length in bit (including pre-processed bytes).
        uint64_t L = 8 * (len + pre_len);
        /// total number blocks to digest.
        uint64_t blk_num = (len >> 6) + 1 + ((len & 0x3f) > 55);
        // inform digest function.
        nblk_strm.write(blk_num);
        end_nblk_strm.write(false);

    LOOP_SHA256_GEN_FULL_BLKS:
        for (uint64_t j = 0; j < uint64_t(len >> 6); ++j) {
#pragma HLS loop_tripcount min = 1 max = 1
            /// message block.
            SHA256Block b0;
#pragma HLS array_partition variable = b0.M complete
        // this block will hold 64 byte of message.
        LOOP_SHA256_GEN_ONE_FULL_BLK:
            for (int i = 0; i < 16; ++i) {
#pragma HLS pipeline II = 1
                uint32_t l = msg_strm.read();
                // XXX algorithm assumes big-endian.
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                b0.M[i] = l;
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (32bx16)\n", i, b0.M[i]);
            }
            // send block
            blk_strm.write(b0);
            _XF_SECURITY_PRINT("DEBUG: block sent\n");
            // shift the buffer. high will be zero.
        }

        /// number of bytes not in blocks yet.
        char left = (char)(len & 0x3fULL); // < 64

        _XF_SECURITY_PRINT("DEBUG: sent = %d, left = %d\n", int(len & (-1ULL ^ 0x3fULL)), (int)left);

        if (left == 0) {
            // end at block boundary, start with pad 1.

            /// last block
            SHA256Block b;
#pragma HLS array_partition variable = b.M complete
            // pad 1
            b.M[0] = 0x80000000UL;
            _XF_SECURITY_PRINT("DEBUG: M[0] =\t%08x (pad 1)\n", b.M[0]);
        // zero
        LOOP_SHA256_GEN_PAD_13_ZEROS:
            for (int i = 1; i < 14; ++i) {
#pragma HLS unroll
                b.M[i] = 0;
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i, b.M[i]);
            }
            // append L
            b.M[14] = (uint32_t)(0xffffffffUL & (L >> 32));
            b.M[15] = (uint32_t)(0xffffffffUL & (L));
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 14, b.M[14]);
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 15, b.M[15]);
            // emit
            blk_strm.write(b);
        } else if (left < 56) {
            // can pad 1 and append L.

            // last message block.
            SHA256Block b;
#pragma HLS array_partition variable = b.M complete

        LOOP_SHA256_GEN_COPY_TAIL_AND_ONE:
            for (int i = 0; i < 14; ++i) {
#pragma HLS pipeline II = 1
                if (i < (left >> 2)) {
                    uint32_t l = msg_strm.read();
                    // pad 1 byte not in this word
                    // XXX algorithm assumes big-endian.
                    l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                        ((0xff000000UL & l) >> 24);
                    b.M[i] = l;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (32b)\n", i, b.M[i]);
                } else if (i > (left >> 2)) {
                    // pad 1 not in this word, and no word to read.
                    b.M[i] = 0UL;
                } else {
                    // pad 1 byte in this word
                    uint32_t e = left & 3L;
                    if (e == 0) {
                        b.M[i] = 0x80000000UL;
                    } else if (e == 1) {
                        uint32_t l = msg_strm.read();
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24);
                        b.M[i] = l | 0x00800000UL;
                    } else if (e == 2) {
                        uint32_t l = msg_strm.read();
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8);
                        b.M[i] = l | 0x00008000UL;
                    } else {
                        uint32_t l = msg_strm.read();
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8);
                        b.M[i] = l | 0x00000080UL;
                    }
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (pad 1)\n", i, b.M[i]);
                }
            }
            // append L
            b.M[14] = (uint32_t)(0xffffffffUL & (L >> 32));
            b.M[15] = (uint32_t)(0xffffffffUL & (L));
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 14, b.M[14]);
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 15, b.M[15]);

            blk_strm.write(b);
            _XF_SECURITY_PRINT("DEBUG: block sent\n");
        } else {
            // cannot append L.

            /// last but 1 block.
            SHA256Block b;
#pragma HLS array_partition variable = b.M complete
        // copy and pad 1
        LOOP_SHA256_GEN_COPY_TAIL_ONLY:
            for (int i = 0; i < 16; ++i) {
#pragma HLS unroll
                if (i < (left >> 2)) {
                    // pad 1 byte not in this word
                    uint32_t l = msg_strm.read();
                    // XXX algorithm assumes big-endian.
                    l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                        ((0xff000000UL & l) >> 24);
                    b.M[i] = l;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (32b)\n", i, b.M[i]);
                } else if (i > (left >> 2)) {
                    // pad 1 byte not in this word, and no msg word to read
                    b.M[i] = 0UL;
                } else {
                    // last in this word
                    uint32_t e = left & 3L;
                    if (e == 0) {
                        b.M[i] = 0x80000000UL;
                    } else if (e == 1) {
                        uint32_t l = msg_strm.read();
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24);
                        b.M[i] = l | 0x00800000UL;
                    } else if (e == 2) {
                        uint32_t l = msg_strm.read();
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8);
                        b.M[i] = l | 0x00008000UL;
                    } else {
                        uint32_t l = msg_strm.read();
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8);
                        b.M[i] = l | 0x00000080UL;
                    }
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (pad 1)\n", i, b.M[i]);
                }
            }
            blk_strm.write(b);
            _XF_SECURITY_PRINT("DEBUG: block sent\n");

            /// last block.
            SHA256Block b1;
#pragma HLS array_partition variable = b1.M complete
        LOOP_SHA256_GEN_L_ONLY_BLK:
            for (int i = 0; i < 14; ++i) {
#pragma HLS unroll
                b1.M[i] = 0;
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i, b1.M[i]);
            }
            // append L
            b1.M[14] = (uint32_t)(0xffffffffUL & (L >> 32));
            b1.M[15] = (uint32_t)(0xffffffffUL & (L));
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 14, b1.M[14]);
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 15, b1.M[15]);

            blk_strm.write(b1);
            _XF_SECURITY_PRINT("DEBUG: block sent\n");
        }
    } // main loop
    end_nblk_strm.write(true);

} // preProcessing (32-bit ver)

/// @brief Generate 512bit processing blocks for SHA224/SHA256 (pipeline)
/// with const width.
/// The performance goal of this function is to yield a 512b block per cycle.
/// @param msg_strm the message being hashed.
/// @param len_strm the message length in byte.
/// @param end_len_strm that flag to signal end of input.
/// @param blk_strm the 512-bit hash block.
/// @param nblk_strm the number of hash block for this message.
/// @param end_nblk_strm end flag for number of hash block.
inline void preProcessing(hls::stream<ap_uint<64> >& msg_strm,
                          hls::stream<ap_uint<64> >& len_strm,
                          hls::stream<bool>& end_len_strm,
                          hls::stream<SHA256Block>& blk_strm,
                          hls::stream<uint64_t>& nblk_strm,
                          hls::stream<bool>& end_nblk_strm) {
LOOP_SHA256_GENENERATE_MAIN:
    for (bool end_flag = end_len_strm.read(); !end_flag; end_flag = end_len_strm.read()) {
        /// message length in byte.
        uint64_t len = len_strm.read();
        _XF_SECURITY_PRINT("DEBUG: working on a new message of %ld bytes\n", len);
        /// message length in bit.
        uint64_t L = 8 * len;
        /// total number blocks to digest.
        uint64_t blk_num = (len >> 6) + 1 + ((len & 0x3f) > 55);
        // inform digest function.
        nblk_strm.write(blk_num);
        end_nblk_strm.write(false);

    LOOP_SHA256_GEN_FULL_BLKS:
        for (uint64_t j = 0; j < uint64_t(len >> 6); ++j) {
#pragma HLS loop_tripcount min = 1 max = 1
            /// message block.
            SHA256Block b0;
#pragma HLS array_partition variable = b0.M complete

        // this block will hold 64 byte of message.
        LOOP_SHA256_GEN_ONE_FULL_BLK:
            for (int i = 0; i < 16; i += 2) {
#pragma HLS pipeline II = 1
                uint64_t ll = msg_strm.read().to_uint64();
                // low
                uint32_t l = ll & 0xffffffffUL;
                // XXX algorithm assumes big-endian.
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                b0.M[i] = l;
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (64bx8 low)\n", i, b0.M[i]);
                // high
                l = (ll >> 32) & 0xffffffffUL;
                // XXX algorithm assumes big-endian.
                l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                    ((0xff000000UL & l) >> 24);
                b0.M[i + 1] = l;
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (64bx8 high)\n", i, b0.M[i]);
            }
            // send block
            blk_strm.write(b0);
            _XF_SECURITY_PRINT("DEBUG: block sent\n");
            // shift the buffer. high will be zero.
        }

        /// number of bytes not in blocks yet.
        char left = (char)(len & 0x3fULL); // < 64

        _XF_SECURITY_PRINT("DEBUG: sent = %d, left = %d\n", int(len & (-1ULL ^ 0x3fULL)), (int)left);

        if (left == 0) {
            // end at block boundary, start with pad 1.

            /// last block
            SHA256Block b;
#pragma HLS array_partition variable = b.M complete
            // pad 1
            b.M[0] = 0x80000000UL;
            _XF_SECURITY_PRINT("DEBUG: M[0] =\t%08x (pad 1)\n", b.M[0]);
        // zero
        LOOP_SHA256_GEN_PAD_13_ZEROS:
            for (int i = 1; i < 14; ++i) {
#pragma HLS unroll
                b.M[i] = 0;
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i, b.M[i]);
            }
            // append L
            b.M[14] = (uint32_t)(0xffffffffUL & (L >> 32));
            b.M[15] = (uint32_t)(0xffffffffUL & (L));
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 14, b.M[14]);
            _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 15, b.M[15]);
            // emit
            blk_strm.write(b);
            _XF_SECURITY_PRINT("DEBUG: block sent\n");
        } else {
            // can pad 1 and append L.

            // last message block.
            SHA256Block b;
#pragma HLS array_partition variable = b.M complete

        LOOP_SHA256_GEN_COPY_TAIL_PAD_ONE:
            for (int i = 0; i < ((left < 56) ? 7 : 8); ++i) {
#pragma HLS pipeline II = 1
                if (i < (left >> 3)) {
                    // pad 1 not in this 64b word, and need to copy
                    uint64_t ll = msg_strm.read().to_uint64();
                    // low
                    uint32_t l = ll & 0xffffffffUL;
                    // XXX algorithm assumes big-endian.
                    l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                        ((0xff000000UL & l) >> 24);
                    b.M[i * 2] = l;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (64b low)\n", i * 2, b.M[i * 2]);
                    // high
                    l = (ll >> 32) & 0xffffffffUL;
                    // XXX algorithm assumes big-endian.
                    l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                        ((0xff000000UL & l) >> 24);
                    b.M[i * 2 + 1] = l;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (64b high)\n", i * 2 + 1, b.M[i * 2 + 1]);
                } else if (i > (left >> 3)) {
                    // pad 1 not in this 64b word, and no word to read.
                    b.M[i * 2] = 0UL;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i * 2, b.M[i * 2]);
                    b.M[i * 2 + 1] = 0UL;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i * 2 + 1, b.M[i * 2 + 1]);
                } else {
                    // pad 1 byte in this 64b word
                    if ((left & 4) == 0) {
                        // left in low 32b
                        uint32_t e = left & 3L;
                        if (e == 0) {
                            b.M[i * 2] = 0x80000000UL;
                        } else if (e == 1) {
                            uint32_t l = msg_strm.read().to_uint64() & 0xffffffffUL;
                            // XXX algorithm assumes big-endian.
                            l = ((0x000000ffUL & l) << 24);
                            b.M[i * 2] = l | 0x00800000UL;
                        } else if (e == 2) {
                            uint32_t l = msg_strm.read().to_uint64() & 0xffffffffUL;
                            // XXX algorithm assumes big-endian.
                            l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8);
                            b.M[i * 2] = l | 0x00008000UL;
                        } else {
                            uint32_t l = msg_strm.read().to_uint64() & 0xffffffffUL;
                            // XXX algorithm assumes big-endian.
                            l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8);
                            b.M[i * 2] = l | 0x00000080UL;
                        }
                        _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (pad 1)\n", i * 2, b.M[i * 2]);
                        // high
                        b.M[i * 2 + 1] = 0UL;
                        _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i * 2 + 1, b.M[i * 2 + 1]);
                    } else {
                        // left in high 32b
                        uint64_t ll = msg_strm.read().to_uint64();
                        // low 32b
                        uint32_t l = ll & 0xffffffffUL;
                        // XXX algorithm assumes big-endian.
                        l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8) |
                            ((0xff000000UL & l) >> 24);
                        b.M[i * 2] = l;
                        _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (64b low)\n", i, b.M[i * 2]);
                        // high 32b
                        l = (ll >> 32) & 0xffffffffUL;
                        uint32_t e = left & 3L;
                        if (e == 0) {
                            b.M[i * 2 + 1] = 0x80000000UL;
                        } else if (e == 1) {
                            // XXX algorithm assumes big-endian.
                            l = ((0x000000ffUL & l) << 24);
                            b.M[i * 2 + 1] = l | 0x00800000UL;
                        } else if (e == 2) {
                            // XXX algorithm assumes big-endian.
                            l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8);
                            b.M[i * 2 + 1] = l | 0x00008000UL;
                        } else {
                            // XXX algorithm assumes big-endian.
                            l = ((0x000000ffUL & l) << 24) | ((0x0000ff00UL & l) << 8) | ((0x00ff0000UL & l) >> 8);
                            b.M[i * 2 + 1] = l | 0x00000080UL;
                        }
                        _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (pad 1)\n", i * 2 + 1, b.M[i * 2 + 1]);
                    }
                }
            }

            if (left < 56) {
                // append L
                b.M[14] = (uint32_t)(0xffffffffUL & (L >> 32));
                b.M[15] = (uint32_t)(0xffffffffUL & (L));
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 14, b.M[14]);
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 15, b.M[15]);

                blk_strm.write(b);
                _XF_SECURITY_PRINT("DEBUG: block sent\n");
            } else {
                // send block without L
                blk_strm.write(b);
                _XF_SECURITY_PRINT("DEBUG: block sent\n");

                /// last block.
                SHA256Block b1;
#pragma HLS array_partition variable = b1.M complete
            LOOP_SHA256_GEN_L_ONLY_BLK:
                for (int i = 0; i < 14; ++i) {
#pragma HLS unroll
                    b1.M[i] = 0;
                    _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (zero)\n", i, b1.M[i]);
                }
                // append L
                b1.M[14] = (uint32_t)(0xffffffffUL & (L >> 32));
                b1.M[15] = (uint32_t)(0xffffffffUL & (L));
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 14, b1.M[14]);
                _XF_SECURITY_PRINT("DEBUG: M[%d] =\t%08x (append L)\n", 15, b1.M[15]);

                blk_strm.write(b1);
                _XF_SECURITY_PRINT("DEBUG: block sent\n");
            } // left < 56
        }
    } // main loop
    end_nblk_strm.write(true);

} // preProcessing (64bit ver)

inline void dup_strm(hls::stream<uint64_t>& in_strm,
                     hls::stream<bool>& in_e_strm,
                     hls::stream<uint64_t>& out1_strm,
                     hls::stream<bool>& out1_e_strm,
                     hls::stream<uint64_t>& out2_strm,
                     hls::stream<bool>& out2_e_strm) {
    bool e = in_e_strm.read();

    while (!e) {
#pragma HLS loop_tripcount min = 1 max = 1 avg = 1
#pragma HLS pipeline II = 1
        uint64_t in_r = in_strm.read();

        out1_strm.write(in_r);
        out1_e_strm.write(false);
        out2_strm.write(in_r);
        out2_e_strm.write(false);

        e = in_e_strm.read();
    }

    out1_e_strm.write(true);
    out2_e_strm.write(true);
}

inline void generateMsgSchedule(hls::stream<SHA256Block>& blk_strm,
                                hls::stream<uint64_t>& nblk_strm,
                                hls::stream<bool>& end_nblk_strm,
                                hls::stream<uint32_t>& w_strm) {
    bool e = end_nblk_strm.read();
    while (!e) {
        uint64_t n = nblk_strm.read();
    LOOP_SHA256_SCHEDULE_BLOCKS:
        for (uint64_t blk_idx = 0; blk_idx < n; ++blk_idx) {
#pragma HLS loop_tripcount min = 2 max = 2
            SHA256Block blk = blk_strm.read();
#pragma HLS array_partition variable = blk.M complete

            uint32_t staged_blk[16];
#pragma HLS array_partition variable = staged_blk complete
        LOOP_SHA256_STAGE_COPY:
            for (int u = 0; u < 16; ++u) {
#pragma HLS unroll
                staged_blk[u] = blk.M[u];
            }
#pragma HLS latency min = 1 max = 1

            uint32_t W16[16];
#pragma HLS array_partition variable = W16 complete

        LOOP_SHA256_SCHEDULE:
            for (int t = 0; t < 64; ++t) {
#pragma HLS pipeline II = 1
                uint32_t Wt;
                if (t < 16) {
                    Wt = staged_blk[t];
                } else {
                    int idx_tm2 = (t - 2) & 0xF;
                    int idx_tm7 = (t - 7) & 0xF;
                    int idx_tm15 = (t - 15) & 0xF;
                    int idx_t = t & 0xF;
                    uint32_t w_tm2 = W16[idx_tm2];
                    uint32_t w_tm7 = W16[idx_tm7];
                    uint32_t w_tm15 = W16[idx_tm15];
                    uint32_t w_tm16 = W16[idx_t];
                    ap_uint<33> sum_a = (ap_uint<33>)SSIG0(w_tm15) + SSIG1(w_tm2);
                    ap_uint<33> sum_b = (ap_uint<33>)w_tm7 + w_tm16;
                    ap_uint<34> sum_ab = (ap_uint<34>)sum_b + (ap_uint<34>)sum_a;
                    Wt = (uint32_t)sum_ab;
                }
                W16[t & 0xF] = Wt;
                w_strm.write(Wt);
            }
        }
        e = end_nblk_strm.read();
    }
}

template <int dataW>
inline void generateOuterScheduleFromDigest(hls::stream<ap_uint<dataW> >& digestWordStrm,
                                            hls::stream<ap_uint<64> >& digestLenStrm,
                                            hls::stream<bool>& eDigestLenStrm,
                                            hls::stream<uint32_t>& w_strm,
                                            hls::stream<uint64_t>& nblk_strm,
                                            hls::stream<bool>& eNblkStrm,
                                            uint64_t pre_len_bytes) {
    XF_SECURITY_STATIC_ASSERT(dataW == 32, "generateOuterScheduleFromDigest expects 32-bit digest words");
    bool done = false;
OUTER_SCHEDULE_MAIN_LOOP:
    while (!done) {
#pragma HLS loop_tripcount min = 1 max = 1
        bool digest_end = eDigestLenStrm.read();
        if (digest_end) {
            eNblkStrm.write(true);
            done = true;
        } else {
            uint64_t digest_len_bytes = digestLenStrm.read();
            uint64_t total_bits = 8 * (pre_len_bytes + digest_len_bytes);
            uint32_t total_bits_hi = (uint32_t)(total_bits >> 32);
            uint32_t total_bits_lo = (uint32_t)(total_bits & 0xffffffffUL);

            nblk_strm.write((uint64_t)1);
            eNblkStrm.write(false);

            uint32_t W16[16];
#pragma HLS array_partition variable = W16 complete

        OUTER_SCHEDULE_PIPELINE_LOOP:
            for (short t = 0; t < 64; ++t) {
#pragma HLS PIPELINE II = 1
                uint32_t Wt;
                if (t < 8) {
                    ap_uint<dataW> w_in = digestWordStrm.read();
                    uint32_t w_raw = (uint32_t)w_in;
                    uint32_t w_be = ((w_raw & 0x000000ffU) << 24) | ((w_raw & 0x0000ff00U) << 8) |
                                    ((w_raw & 0x00ff0000U) >> 8) | ((w_raw & 0xff000000U) >> 24);
                    Wt = w_be;
                } else if (t == 8) {
                    Wt = 0x80000000u;
                } else if (t < 14) {
                    Wt = 0u;
                } else if (t == 14) {
                    Wt = total_bits_hi;
                } else if (t == 15) {
                    Wt = total_bits_lo;
                } else {
                    uint32_t w_tm2 = W16[(t - 2) & 0xF];
                    uint32_t w_tm7 = W16[(t - 7) & 0xF];
                    uint32_t w_tm15 = W16[(t - 15) & 0xF];
                    uint32_t w_tm16 = W16[t & 0xF];
                    ap_uint<33> sum_a = (ap_uint<33>)SSIG0(w_tm15) + SSIG1(w_tm2);
                    ap_uint<33> sum_b = (ap_uint<33>)w_tm7 + w_tm16;
                    ap_uint<34> sum_ab = (ap_uint<34>)sum_b + (ap_uint<34>)sum_a;
                    Wt = (uint32_t)sum_ab;
                }
                W16[t & 0xF] = Wt;
                w_strm.write(Wt);
            }
        }
    }
}


template <int hshW>
inline void outerScheduleFromDigestStage(hls::stream<ap_uint<hshW> >& digestStrm,
                                         hls::stream<bool>& eDigestStrm,
                                         hls::stream<uint32_t>& w_strm,
                                         uint64_t pre_len_bytes) {
#pragma HLS INLINE off
    XF_SECURITY_STATIC_ASSERT((hshW % 32) == 0, "Digest width must be a multiple of 32 bits");
    const int words = hshW / 32;

    bool done = false;
OUTER_FAST_MAIN_LOOP:
    while (!done) {
#pragma HLS loop_tripcount min = 1 max = 1
        bool digest_end = eDigestStrm.read();
        if (digest_end) {
            done = true;
        } else {
            ap_uint<hshW> digest = digestStrm.read();

            uint32_t initial_words[8];
#pragma HLS array_partition variable = initial_words complete
        OUTER_FAST_DIGEST_LOAD:
            for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
                uint32_t raw = (uint32_t)digest.range(32 * (i + 1) - 1, 32 * i);
                initial_words[i] = ((raw & 0x000000ffU) << 24) | ((raw & 0x0000ff00U) << 8) |
                                   ((raw & 0x00ff0000UL) >> 8) | ((raw & 0xff000000UL) >> 24);
            }

            uint64_t total_bits = 8 * (pre_len_bytes + (hshW / 8));
            uint32_t total_bits_hi = (uint32_t)(total_bits >> 32);
            uint32_t total_bits_lo = (uint32_t)(total_bits & 0xffffffffUL);

            uint32_t W16[16];
#pragma HLS array_partition variable = W16 complete

        OUTER_FAST_SCHEDULE_LOOP:
            for (int t = 0; t < 64; ++t) {
#pragma HLS PIPELINE II = 1
                uint32_t Wt;
                if (t < words) {
                    Wt = initial_words[t];
                } else if (t == 8) {
                    Wt = 0x80000000u;
                } else if (t < 14) {
                    Wt = 0u;
                } else if (t == 14) {
                    Wt = total_bits_hi;
                } else if (t == 15) {
                    Wt = total_bits_lo;
                } else {
                    int idx_tm2 = (t - 2) & 0xF;
                    int idx_tm7 = (t - 7) & 0xF;
                    int idx_tm15 = (t - 15) & 0xF;
                    int idx_t = t & 0xF;
                    uint32_t w_tm2 = W16[idx_tm2];
                    uint32_t w_tm7 = W16[idx_tm7];
                    uint32_t w_tm15 = W16[idx_tm15];
                    uint32_t w_tm16 = W16[idx_t];
                    ap_uint<33> sum_a = (ap_uint<33>)SSIG0(w_tm15) + SSIG1(w_tm2);
                    ap_uint<33> sum_b = (ap_uint<33>)w_tm7 + w_tm16;
                    ap_uint<34> sum_ab = (ap_uint<34>)sum_b + (ap_uint<34>)sum_a;
                    Wt = (uint32_t)sum_ab;
                }
                W16[t & 0xF] = Wt;
                w_strm.write(Wt);
            }
        }
    }
}

inline void sha256_iter(uint32_t& a,
                        uint32_t& b,
                        uint32_t& c,
                        uint32_t& d,
                        uint32_t& e,
                        uint32_t& f,
                        uint32_t& g,
                        uint32_t& h,
                        hls::stream<uint32_t>& w_strm,
                        uint32_t& Kt,
                        const uint32_t K[],
                        int t) {
    uint32_t Wt = w_strm.read();
    /// temporal variables
    uint32_t sig1 = BSIG1(e);
    uint32_t sig1_for_t1 = sig1;
#pragma HLS RESOURCE variable = sig1_for_t1 core = Register
    uint32_t ch_val = CH(e, f, g);
    uint32_t ch_val_for_t1 = ch_val;
#pragma HLS RESOURCE variable = ch_val_for_t1 core = Register
    uint32_t sig0 = BSIG0(a);
    uint32_t sig0_for_t2 = sig0;
#pragma HLS RESOURCE variable = sig0_for_t2 core = Register
    uint32_t maj_val = MAJ(a, b, c);
    uint32_t maj_val_for_t2 = maj_val;
#pragma HLS RESOURCE variable = maj_val_for_t2 core = Register
    ap_uint<33> t1_part0 = (ap_uint<33>)h + sig1_for_t1;
    ap_uint<33> t1_part1 = (ap_uint<33>)ch_val_for_t1 + Wt;
    ap_uint<34> t1_mid = (ap_uint<34>)t1_part0 + (ap_uint<34>)t1_part1;
    ap_uint<35> T1_wide = t1_mid + (ap_uint<35>)Kt;
    ap_uint<33> T2_wide = (ap_uint<33>)sig0_for_t2 + maj_val_for_t2;
    uint32_t T1 = (uint32_t)T1_wide;
    uint32_t T2 = (uint32_t)T2_wide;

    // update working variables.
    h = g;
    g = f;
    f = e;
    e = d + T1;
    d = c;
    c = b;
    b = a;
    a = T1 + T2;

    _XF_SECURITY_PRINT(
        "DEBUG: Kt=%08x, Wt=%08x\n"
        "\ta=%08x, b=%08x, c=%08x, d=%08x\n"
        "\te=%08x, f=%08x, g=%08x, h=%08x\n",
        Kt, Wt, a, b, c, d, e, f, g, h);

    // for next cycle
    Kt = K[(t + 1) & 63];
}
/// @brief Digest message blocks and emit final hash.
/// @tparam h_width the hash width(type).
/// @param nblk_strm number of message block.
/// @param end_nblk_strm end flag for number of message block.
/// @param hash_strm the hash result stream.
/// @param end_hash_strm end flag for hash result.
template <int h_width>
void sha256Digest(hls::stream<uint64_t>& nblk_strm,
                  hls::stream<bool>& end_nblk_strm,
                  hls::stream<uint32_t>& w_strm,
                  hls::stream<ap_uint<h_width> >& hash_strm,
                  hls::stream<bool>& end_hash_strm) {
    // h_width determine the hash type.
    XF_SECURITY_STATIC_ASSERT((h_width == 256) || (h_width == 224),
                              "Unsupported hash stream width, must be 224 or 256");

    /// constant K
#pragma HLS array_partition variable = SHA256_K complete
#pragma HLS bind_storage variable = SHA256_K type = rom_1p impl = lutram
#pragma HLS array_partition variable = SHA256_IV256 complete
#pragma HLS bind_storage variable = SHA256_IV256 type = rom_1p impl = lutram
#pragma HLS array_partition variable = SHA256_IV224 complete
#pragma HLS bind_storage variable = SHA256_IV224 type = rom_1p impl = lutram

LOOP_SHA256_DIGEST_MAIN:
    for (bool end_flag = end_nblk_strm.read(); !end_flag; end_flag = end_nblk_strm.read()) {
        /// total number blocks to digest.
        uint64_t blk_num = nblk_strm.read();
        // _XF_SECURITY_PRINT("expect %ld blocks.\n", blk_num);

        /// internal states
        uint32_t H[8];
#pragma HLS array_partition variable = H complete

        // initialize
        if (h_width == 224) {
        LOOP_SHA256_LOAD_IV224:
            for (short i = 0; i < 8; ++i) {
#pragma HLS unroll
                H[i] = SHA256_IV224[i];
            }
        } else {
        LOOP_SHA256_LOAD_IV256:
            for (short i = 0; i < 8; ++i) {
#pragma HLS unroll
                H[i] = SHA256_IV256[i];
            }
        }

    LOOP_SHA256_DIGEST_NBLK:
        for (uint64_t n = 0; n < blk_num; ++n) {
#pragma HLS loop_tripcount min = 1 max = 1
#pragma HLS latency max = 65

            /// working variables.
            uint32_t a, b, c, d, e, f, g, h;

            // loading working variables.
            a = H[0];
            b = H[1];
            c = H[2];
            d = H[3];
            e = H[4];
            f = H[5];
            g = H[6];
            h = H[7];

            uint32_t Kt = SHA256_K[0];
        LOOP_SHA256_UPDATE_64_ROUNDS:
            for (int t = 0; t < 64; ++t) {
#pragma HLS pipeline II = 1
                sha256_iter(a, b, c, d, e, f, g, h, w_strm, Kt, SHA256_K, t);
            } // 64 round loop

            // store working variables to internal states.
            H[0] = a + H[0];
            H[1] = b + H[1];
            H[2] = c + H[2];
            H[3] = d + H[3];
            H[4] = e + H[4];
            H[5] = f + H[5];
            H[6] = g + H[6];
            H[7] = h + H[7];
        } // block loop

        // Emit digest
        if (h_width == 224) {
            ap_uint<224> w224;
        LOOP_SHA256_EMIT_H224:
            for (short i = 0; i < sha256_digest_config<true>::numH; ++i) {
#pragma HLS unroll
                uint32_t l = H[i];
                // XXX shift algorithm's big endian to HLS's little endian.
                uint8_t t0 = (((l) >> 24) & 0xff);
                uint8_t t1 = (((l) >> 16) & 0xff);
                uint8_t t2 = (((l) >> 8) & 0xff);
                uint8_t t3 = (((l)) & 0xff);
                uint32_t l_little =
                    ((uint32_t)t0) | (((uint32_t)t1) << 8) | (((uint32_t)t2) << 16) | (((uint32_t)t3) << 24);
                w224.range(32 * i + 31, 32 * i) = l_little;
            }
            end_hash_strm.write(false);
            hash_strm.write(w224);
        } else {
            ap_uint<256> w256;
        LOOP_SHA256_EMIT_H256:
            for (short i = 0; i < sha256_digest_config<false>::numH; ++i) {
#pragma HLS unroll
                uint32_t l = H[i];
                // XXX shift algorithm's big endian to HLS's little endian.
                uint8_t t0 = (((l) >> 24) & 0xff);
                uint8_t t1 = (((l) >> 16) & 0xff);
                uint8_t t2 = (((l) >> 8) & 0xff);
                uint8_t t3 = (((l)) & 0xff);
                uint32_t l_little =
                    ((uint32_t)t0) | (((uint32_t)t1) << 8) | (((uint32_t)t2) << 16) | (((uint32_t)t3) << 24);
                w256.range(32 * i + 31, 32 * i) = l_little;
            }
            end_hash_strm.write(false);
            hash_strm.write(w256);
        }
    } // main loop
    end_hash_strm.write(true);

} // sha256Digest (pipelined override)

template <int h_width>
void sha256DigestWithState(hls::stream<uint64_t>& nblk_strm,
                           hls::stream<bool>& end_nblk_strm,
                           hls::stream<uint32_t>& w_strm,
                           hls::stream<ap_uint<256> >& init_state_strm,
                           hls::stream<bool>& end_state_strm,
                           hls::stream<ap_uint<h_width> >& hash_strm,
                           hls::stream<bool>& end_hash_strm) {
#pragma HLS array_partition variable = SHA256_K complete
#pragma HLS bind_storage variable = SHA256_K type = rom_1p impl = lutram

    bool done = false;
LOOP_SHA256_DIGEST_STATE_MAIN:
    while (!done) {
        bool end_flag = end_nblk_strm.read();
        bool end_state = end_state_strm.read();
        if (end_flag || end_state) {
            end_hash_strm.write(true);
            done = true;
        } else {
            uint64_t blk_num = nblk_strm.read();
            ap_uint<256> packed_state = init_state_strm.read();

        uint32_t H[8];
#pragma HLS array_partition variable = H complete
        unpackState(packed_state, H);

    LOOP_SHA256_DIGEST_STATE_NBLK:
        for (uint64_t n = 0; n < blk_num; ++n) {
#pragma HLS loop_tripcount min = 2 max = 2
#pragma HLS latency max = 65
            uint32_t a = H[0];
            uint32_t b = H[1];
            uint32_t c = H[2];
            uint32_t d = H[3];
            uint32_t e = H[4];
            uint32_t f = H[5];
            uint32_t g = H[6];
            uint32_t h = H[7];

            uint32_t Kt = SHA256_K[0];
        LOOP_SHA256_UPDATE_64_ROUNDS_STATE:
            for (int t = 0; t < 64; ++t) {
#pragma HLS pipeline II = 1
                sha256_iter(a, b, c, d, e, f, g, h, w_strm, Kt, SHA256_K, t);
            }

            H[0] = a + H[0];
            H[1] = b + H[1];
            H[2] = c + H[2];
            H[3] = d + H[3];
            H[4] = e + H[4];
            H[5] = f + H[5];
            H[6] = g + H[6];
            H[7] = h + H[7];
        }

        if (h_width == 224) {
            ap_uint<224> w224;
        LOOP_SHA256_EMIT_STATE_H224:
            for (short i = 0; i < sha256_digest_config<true>::numH; ++i) {
#pragma HLS unroll
                uint32_t l = H[i];
                uint8_t t0 = (((l) >> 24) & 0xff);
                uint8_t t1 = (((l) >> 16) & 0xff);
                uint8_t t2 = (((l) >> 8) & 0xff);
                uint8_t t3 = (((l)) & 0xff);
                uint32_t l_little =
                    ((uint32_t)t0) | (((uint32_t)t1) << 8) | (((uint32_t)t2) << 16) | (((uint32_t)t3) << 24);
                w224.range(32 * i + 31, 32 * i) = l_little;
            }
            end_hash_strm.write(false);
            hash_strm.write(w224);
        } else {
            ap_uint<256> w256;
        LOOP_SHA256_EMIT_STATE_H256:
            for (short i = 0; i < sha256_digest_config<false>::numH; ++i) {
#pragma HLS unroll
                uint32_t l = H[i];
                uint8_t t0 = (((l) >> 24) & 0xff);
                uint8_t t1 = (((l) >> 16) & 0xff);
                uint8_t t2 = (((l) >> 8) & 0xff);
                uint8_t t3 = (((l)) & 0xff);
                uint32_t l_little =
                    ((uint32_t)t0) | (((uint32_t)t1) << 8) | (((uint32_t)t2) << 16) | (((uint32_t)t3) << 24);
                w256.range(32 * i + 31, 32 * i) = l_little;
            }
            end_hash_strm.write(false);
            hash_strm.write(w256);
        }
    }
    }
}

template <int h_width>
void sha256DigestOneBlockWithState(hls::stream<uint32_t>& w_strm,
                                   hls::stream<ap_uint<256> >& init_state_strm,
                                   hls::stream<bool>& end_state_strm,
                                   hls::stream<ap_uint<h_width> >& hash_strm,
                                   hls::stream<bool>& end_hash_strm) {
#pragma HLS array_partition variable = SHA256_K complete
#pragma HLS bind_storage variable = SHA256_K type = rom_1p impl = lutram

    bool done = false;
LOOP_SHA256_DIGEST_ONEBLOCK_MAIN:
    while (!done) {
        bool end_state = end_state_strm.read();
        if (end_state) {
            end_hash_strm.write(true);
            done = true;
        } else {
            ap_uint<256> packed_state = init_state_strm.read();

            uint32_t H[8];
#pragma HLS array_partition variable = H complete
            unpackState(packed_state, H);

            uint32_t a = H[0];
            uint32_t b = H[1];
            uint32_t c = H[2];
            uint32_t d = H[3];
            uint32_t e = H[4];
            uint32_t f = H[5];
            uint32_t g = H[6];
            uint32_t h = H[7];

            uint32_t Kt = SHA256_K[0];
        LOOP_SHA256_DIGEST_ONEBLOCK:
            for (int t = 0; t < 64; ++t) {
#pragma HLS pipeline II = 1
                sha256_iter(a, b, c, d, e, f, g, h, w_strm, Kt, SHA256_K, t);
            }

            H[0] = a + H[0];
            H[1] = b + H[1];
            H[2] = c + H[2];
            H[3] = d + H[3];
            H[4] = e + H[4];
            H[5] = f + H[5];
            H[6] = g + H[6];
            H[7] = h + H[7];

            if (h_width == 224) {
                ap_uint<224> w224;
            LOOP_SHA256_EMIT_ONEBLOCK_H224:
                for (short i = 0; i < sha256_digest_config<true>::numH; ++i) {
#pragma HLS unroll
                    uint32_t l = H[i];
                    uint8_t t0 = (((l) >> 24) & 0xff);
                    uint8_t t1 = (((l) >> 16) & 0xff);
                    uint8_t t2 = (((l) >> 8) & 0xff);
                    uint8_t t3 = (((l)) & 0xff);
                    uint32_t l_little =
                        ((uint32_t)t0) | (((uint32_t)t1) << 8) | (((uint32_t)t2) << 16) | (((uint32_t)t3) << 24);
                    w224.range(32 * i + 31, 32 * i) = l_little;
                }
                end_hash_strm.write(false);
                hash_strm.write(w224);
            } else {
                ap_uint<256> w256;
            LOOP_SHA256_EMIT_ONEBLOCK_H256:
                for (short i = 0; i < sha256_digest_config<false>::numH; ++i) {
#pragma HLS unroll
                    uint32_t l = H[i];
                    uint8_t t0 = (((l) >> 24) & 0xff);
                    uint8_t t1 = (((l) >> 16) & 0xff);
                    uint8_t t2 = (((l) >> 8) & 0xff);
                    uint8_t t3 = (((l)) & 0xff);
                    uint32_t l_little =
                        ((uint32_t)t0) | (((uint32_t)t1) << 8) | (((uint32_t)t2) << 16) | (((uint32_t)t3) << 24);
                    w256.range(32 * i + 31, 32 * i) = l_little;
                }
                end_hash_strm.write(false);
                hash_strm.write(w256);
            }
        }
    }
}

/// @brief SHA-256/224 implementation top overload for ap_uint input.
/// @tparam m_width the input message stream width.
/// @tparam h_width the output hash stream width.
/// @param msg_strm the message being hashed.
/// @param len_strm the length message being hashed in byte.
/// @param end_len_strm end flag stream of input, one per message.
/// @param hash_strm the result.
/// @param end_hash_strm end falg stream of output, one per hash.
template <int m_width, int h_width>
inline void sha256_top(hls::stream<ap_uint<m_width> >& msg_strm,
                       hls::stream<ap_uint<64> >& len_strm,
                       hls::stream<bool>& end_len_strm,
                       hls::stream<ap_uint<h_width> >& hash_strm,
                       hls::stream<bool>& end_hash_strm) {
#pragma HLS DATAFLOW disable_start_propagation
    /// 512-bit Block stream
    hls::stream<SHA256Block> blk_strm("blk_strm");
#pragma HLS STREAM variable = blk_strm depth = 4
#pragma HLS RESOURCE variable = blk_strm core = FIFO_LUTRAM

    /// number of Blocks, send per msg
    hls::stream<uint64_t> nblk_strm("nblk_strm");
#pragma HLS STREAM variable = nblk_strm depth = 2
#pragma HLS RESOURCE variable = nblk_strm core = FIFO_SRL
    hls::stream<uint64_t> nblk_strm1("nblk_strm1");
#pragma HLS STREAM variable = nblk_strm1 depth = 2
#pragma HLS RESOURCE variable = nblk_strm1 core = FIFO_SRL
    hls::stream<uint64_t> nblk_strm2("nblk_strm2");
#pragma HLS STREAM variable = nblk_strm2 depth = 2
#pragma HLS RESOURCE variable = nblk_strm2 core = FIFO_SRL

    /// end flag, send per msg.
    hls::stream<bool> end_nblk_strm("end_nblk_strm");
#pragma HLS STREAM variable = end_nblk_strm depth = 2
#pragma HLS RESOURCE variable = end_nblk_strm core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm1("end_nblk_strm1");
#pragma HLS STREAM variable = end_nblk_strm1 depth = 2
#pragma HLS RESOURCE variable = end_nblk_strm1 core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm2("end_nblk_strm2");
#pragma HLS STREAM variable = end_nblk_strm2 depth = 2
#pragma HLS RESOURCE variable = end_nblk_strm2 core = FIFO_LUTRAM

    /// W, 64 items for each block
    hls::stream<uint32_t> w_strm("w_strm");
#pragma HLS STREAM variable = w_strm depth = 128
#pragma HLS RESOURCE variable = w_strm core = FIFO_LUTRAM

    // Generate block stream
    preProcessing(msg_strm, len_strm, end_len_strm, //
                  blk_strm, nblk_strm, end_nblk_strm);

    // Duplicate number of block stream and its end flag stream
    dup_strm(nblk_strm, end_nblk_strm, nblk_strm1, end_nblk_strm1, nblk_strm2, end_nblk_strm2);

    // Generate the message schedule in stream
    generateMsgSchedule(blk_strm, nblk_strm1, end_nblk_strm1, w_strm);

    // Digest block stream, and write hash stream.
    // fully pipelined version will calculate SHA-224 if hash_strm width is 224.
    sha256Digest(nblk_strm2, end_nblk_strm2, w_strm, //
                 hash_strm, end_hash_strm);
} // sha256_top

template <int m_width, int h_width>
inline void sha256_top_with_state(hls::stream<ap_uint<m_width> >& msg_strm,
                                  hls::stream<ap_uint<64> >& len_strm,
                                  hls::stream<bool>& end_len_strm,
                                  hls::stream<ap_uint<256> >& init_state_strm,
                                  hls::stream<bool>& end_state_strm,
                                  hls::stream<ap_uint<h_width> >& hash_strm,
                                  hls::stream<bool>& end_hash_strm,
                                  uint64_t pre_len) {
#pragma HLS DATAFLOW disable_start_propagation
    hls::stream<SHA256Block> blk_strm("blk_strm_state");
#pragma HLS STREAM variable = blk_strm depth = 32
#pragma HLS RESOURCE variable = blk_strm core = FIFO_LUTRAM

    hls::stream<uint64_t> nblk_strm("nblk_strm_state");
#pragma HLS STREAM variable = nblk_strm depth = 16
#pragma HLS RESOURCE variable = nblk_strm core = FIFO_LUTRAM
    hls::stream<uint64_t> nblk_strm1("nblk_strm_state_1");
#pragma HLS STREAM variable = nblk_strm1 depth = 16
#pragma HLS RESOURCE variable = nblk_strm1 core = FIFO_LUTRAM
    hls::stream<uint64_t> nblk_strm2("nblk_strm_state_2");
#pragma HLS STREAM variable = nblk_strm2 depth = 16
#pragma HLS RESOURCE variable = nblk_strm2 core = FIFO_LUTRAM

    hls::stream<bool> end_nblk_strm("end_nblk_strm_state");
#pragma HLS STREAM variable = end_nblk_strm depth = 2
#pragma HLS RESOURCE variable = end_nblk_strm core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm1("end_nblk_strm_state_1");
#pragma HLS STREAM variable = end_nblk_strm1 depth = 2
#pragma HLS RESOURCE variable = end_nblk_strm1 core = FIFO_LUTRAM
    hls::stream<bool> end_nblk_strm2("end_nblk_strm_state_2");
#pragma HLS STREAM variable = end_nblk_strm2 depth = 2
#pragma HLS RESOURCE variable = end_nblk_strm2 core = FIFO_LUTRAM

    hls::stream<uint32_t> w_strm("w_strm_state");
#pragma HLS STREAM variable = w_strm depth = 32
#pragma HLS RESOURCE variable = w_strm core = FIFO_LUTRAM

    preProcessing(msg_strm, len_strm, end_len_strm, blk_strm, nblk_strm, end_nblk_strm, pre_len);

    dup_strm(nblk_strm, end_nblk_strm, nblk_strm1, end_nblk_strm1, nblk_strm2, end_nblk_strm2);

    generateMsgSchedule(blk_strm, nblk_strm1, end_nblk_strm1, w_strm);

    sha256DigestWithState(nblk_strm2, end_nblk_strm2, w_strm, init_state_strm, end_state_strm, hash_strm, end_hash_strm);
}

} // namespace internal

/// @brief SHA-224 algorithm with ap_uint stream input and output.
/// @tparam m_width the input message stream width, currently only 32 allowed.
/// @param msg_strm the message being hashed.
/// @param len_strm the length message being hashed.
/// @param end_len_strm the flag for end of message length input.
/// @param hash_strm the result.
/// @param end_hash_strm the flag for end of hash output.
template <int m_width>
void sha224(hls::stream<ap_uint<m_width> >& msg_strm,      // in
            hls::stream<ap_uint<64> >& len_strm,           // in
            hls::stream<bool>& end_len_strm,               // in
            hls::stream<ap_uint<224> >& hash_strm,         // out
            hls::stream<bool>& end_hash_strm) {            // out
    internal::sha256_top(msg_strm, len_strm, end_len_strm, // in
                         hash_strm, end_hash_strm);        // out
}

/// @brief SHA-256 algorithm with ap_uint stream input and output.
/// @tparam m_width the input message stream width, currently only 32 allowed.
/// @param msg_strm the message being hashed.
/// @param len_strm the length message being hashed.
/// @param end_len_strm the flag for end of message length input.
/// @param hash_strm the result.
/// @param end_hash_strm the flag for end of hash output.
template <int m_width>
void sha256(hls::stream<ap_uint<m_width> >& msg_strm,      // in
            hls::stream<ap_uint<64> >& len_strm,           // in
            hls::stream<bool>& end_len_strm,               // in
            hls::stream<ap_uint<256> >& hash_strm,         // out
            hls::stream<bool>& end_hash_strm) {            // out
    internal::sha256_top(msg_strm, len_strm, end_len_strm, // in
                         hash_strm, end_hash_strm);        // out
}

template <int m_width>
void sha256_with_state(hls::stream<ap_uint<m_width> >& msg_strm,          // in
                       hls::stream<ap_uint<64> >& len_strm,               // in
                       hls::stream<bool>& end_len_strm,                   // in
                       hls::stream<ap_uint<256> >& init_state_strm,       // in
                       hls::stream<bool>& end_state_strm,                 // in
                       hls::stream<ap_uint<256> >& hash_strm,             // out
                       hls::stream<bool>& end_hash_strm,                  // out
                       uint64_t pre_len_bytes) {                          // processed bytes
#pragma HLS INLINE off
    internal::sha256_top_with_state(msg_strm, len_strm, end_len_strm, init_state_strm, end_state_strm, hash_strm,
                                     end_hash_strm, pre_len_bytes);
}

} // namespace security
} // namespace xf

// Clean up macros.
#undef ROTR
#undef ROTL
#undef SHR
#undef CH
#undef MAJ
#undef BSIG0
#undef BSIG1
#undef SSIG0
#undef SSIG1

#undef _XF_SECURITY_PRINT
#undef _XF_SECURITY_VOID_CAST

#endif // XF_SECURITY_SHA2_H
// -*- cpp -*-
// vim: ts=8:sw=2:sts=2:ft=cpp


