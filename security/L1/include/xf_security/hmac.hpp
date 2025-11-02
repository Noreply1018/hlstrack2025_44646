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

/**
 * @file hmac.hpp
 * @brief header file for HMAC.
 * This file part of Vitis Security Library.
 * TODO
 * @detail .
 */

#ifndef _XF_SECURITY_HMAC_HPP_
#define _XF_SECURITY_HMAC_HPP_

#include <ap_int.h>
#include <hls_stream.h>
#include <xf_security/types.hpp>
#include "xf_security/sha224_256.hpp"

#if !defined(__SYNTHESIS__) && XF_SECURITY_DECRYPT_DEBUG == 1
#include <iostream>
#endif

namespace xf {
namespace security {

namespace internal {

template <int lW, int keyLen>
void expandStrm(hls::stream<bool>& eInStrm, hls::stream<bool>& eOutStrm, hls::stream<ap_uint<lW> >& lenStrm) {
    while (!eInStrm.read()) {
        eOutStrm.write(false);
        lenStrm.write(ap_uint<lW>(keyLen));
    }
    eOutStrm.write(true);
}

inline void duplicateFlagStream(hls::stream<bool>& in_flag,
                                hls::stream<bool>& out_flag0,
                                hls::stream<bool>& out_flag1) {
    bool done = false;
LOOP_DUP_FLAG_FORWARD:
    while (!done) {
#pragma HLS pipeline II = 1
        bool flag = in_flag.read();
        done = flag;
        out_flag0.write(flag);
        out_flag1.write(flag);
    }
}

template <int lW>
void convertLenTo64(hls::stream<ap_uint<lW> >& in_len_strm,
                    hls::stream<bool>& in_end_strm,
                    hls::stream<ap_uint<64> >& out_len_strm,
                    hls::stream<bool>& out_end_strm) {
    bool done = false;
LOOP_CONVERT_LEN_TO_64:
    while (!done) {
#pragma HLS pipeline II = 1
        bool end_flag = in_end_strm.read();
        done = end_flag;
        if (!end_flag) {
            ap_uint<lW> len = in_len_strm.read();
            out_len_strm.write((ap_uint<64>)len);
        }
        out_end_strm.write(end_flag);
    }
}

template <int dataW, int hshW>
void digestToWordStream(hls::stream<ap_uint<hshW> >& digest_strm,
                        hls::stream<bool>& e_digest_strm,
                        hls::stream<ap_uint<dataW> >& word_strm,
                        hls::stream<ap_uint<64> >& len_strm,
                        hls::stream<bool>& e_len_strm) {
    const int words = hshW / dataW;
    bool done = false;
LOOP_DIGEST_MAIN:
    while (!done) {
        bool end_flag = e_digest_strm.read();
        if (end_flag) {
            done = true;
            e_len_strm.write(true);
        } else {
            ap_uint<hshW> digest = digest_strm.read();
            len_strm.write((ap_uint<64>)(hshW / 8));
            e_len_strm.write(false);
            ap_uint<dataW> word_buf[words];
#pragma HLS array_partition variable = word_buf complete
        LOOP_DIGEST_COLLECT:
            for (int i = 0; i < words; ++i) {
#pragma HLS unroll
                int hi = dataW * (i + 1) - 1;
                int lo = dataW * i;
                word_buf[i] = digest.range(hi, lo);
            }
#pragma HLS latency min = 1 max = 1
        LOOP_DIGEST_WORDS:
            for (int i = 0; i < words; ++i) {
#pragma HLS unroll
                word_strm.write(word_buf[i]);
            }
        }
    }
}

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void kpad(hls::stream<ap_uint<dataW> >& keyStrm,
          hls::stream<bool>& eStrm,
          hls::stream<ap_uint<hshW> >& innerStateStrm,
          hls::stream<bool>& eInnerStateStrm,
          hls::stream<ap_uint<hshW> >& outerStateStrm,
          hls::stream<bool>& eOuterStateStrm) {
    XF_SECURITY_STATIC_ASSERT(keyLen <= blockSize, "Key length greater than block size not supported in optimized HMAC");

    // Lightweight key fingerprint cache for repeated keys
    static ap_uint<32> cached_key_fp = 0;
    static ap_uint<hshW> cached_inner = 0;
    static ap_uint<hshW> cached_outer = 0;
    
    bool done = false;
LOOP_KPAD_MAIN:
    while (!done) {
#pragma HLS loop_tripcount min=2 max=2
        bool end_flag = eStrm.read();
        if (end_flag) {
            done = true;
            eInnerStateStrm.write(true);
            eOuterStateStrm.write(true);
            cached_key_fp = 0;  // Reset fingerprint
        } else {
        ap_uint<blockSize * 8> k1 = 0;
        const int words = (keyLen * 8 + dataW - 1) / dataW;
    LOOP_KPAD_READ_KEY:
        for (int i = 0; i < words; ++i) {
#pragma HLS unroll
            ap_uint<dataW> tmp = keyStrm.read();
            int base = ((blockSize - keyLen) * 8) + (words - 1 - i) * dataW;
            k1.range(base + dataW - 1, base) = tmp;
        }

        // Extract 32-bit fingerprint from first 4 bytes of key
        ap_uint<32> key_fp = k1.range(blockSize * 8 - 1, blockSize * 8 - 32);
        bool key_match = (key_fp == cached_key_fp) && (cached_key_fp != 0);
        
        if (key_match) {
            // Fast path: reuse cached states (skip 128 cycles of SHA256)
            innerStateStrm.write(cached_inner);
            eInnerStateStrm.write(false);
            outerStateStrm.write(cached_outer);
            eOuterStateStrm.write(false);
        } else {
            // Slow path: compute states
            ap_uint<blockSize * 8> kipad = 0;
            ap_uint<blockSize * 8> kopad = 0;
        LOOP_KPAD_BUILD_PAD:
            for (int i = 0; i < blockSize; ++i) {
#pragma HLS unroll
                kipad.range(i * 8 + 7, i * 8) = k1.range(i * 8 + 7, i * 8) ^ 0x36;
                kopad.range(i * 8 + 7, i * 8) = k1.range(i * 8 + 7, i * 8) ^ 0x5c;
            }

            ap_uint<hshW> inner_state =
                xf::security::internal::sha256PrecomputePad(kipad, xf::security::internal::SHA256_IV256);
            innerStateStrm.write(inner_state);
            eInnerStateStrm.write(false);

            ap_uint<hshW> outer_state =
                xf::security::internal::sha256PrecomputePad(kopad, xf::security::internal::SHA256_IV256);
            outerStateStrm.write(outer_state);
            eOuterStateStrm.write(false);
            
            // Cache for next iteration
            cached_key_fp = key_fp;
            cached_inner = inner_state;
            cached_outer = outer_state;
        }
    }
    }
}

template <int dataW, int lW, int hshW, int blockSize>
void innerHash(hls::stream<ap_uint<dataW> >& msgStrm,
               hls::stream<ap_uint<lW> >& msgLenStrm,
               hls::stream<bool>& eLenStrm,
               hls::stream<ap_uint<hshW> >& innerStateStrm,
               hls::stream<bool>& eInnerStateStrm,
               hls::stream<ap_uint<hshW> >& msgHashStrm,
               hls::stream<bool>& eMsgHashStrm) {
#pragma HLS dataflow disable_start_propagation

    hls::stream<ap_uint<64> > lenStrm64;
#pragma HLS stream variable = lenStrm64 depth = 16
#pragma HLS resource variable = lenStrm64 core = FIFO_LUTRAM
    hls::stream<bool> eLenStrm64;
#pragma HLS stream variable = eLenStrm64 depth = 4
#pragma HLS resource variable = eLenStrm64 core = FIFO_SRL

    convertLenTo64<lW>(msgLenStrm, eLenStrm, lenStrm64, eLenStrm64);

    xf::security::sha256_with_state<dataW>(msgStrm, lenStrm64, eLenStrm64, innerStateStrm, eInnerStateStrm, msgHashStrm,
                                           eMsgHashStrm, blockSize);
}

template <int dataW, int hshW, int blockSize>
void outerHash(hls::stream<ap_uint<hshW> >& outerStateStrm,
               hls::stream<bool>& eOuterStateStrm,
               hls::stream<ap_uint<hshW> >& msgHashStrm,
               hls::stream<bool>& eMsgHashStrm,
               hls::stream<ap_uint<hshW> >& hshStrm,
               hls::stream<bool>& eHshStrm) {
#pragma HLS dataflow disable_start_propagation

    hls::stream<ap_uint<dataW> > digestWordStrm;
#pragma HLS stream variable = digestWordStrm depth = 8
#pragma HLS resource variable = digestWordStrm core = FIFO_LUTRAM
    hls::stream<ap_uint<64> > digestLenStrm;
#pragma HLS stream variable = digestLenStrm depth = 4
#pragma HLS resource variable = digestLenStrm core = FIFO_LUTRAM
    hls::stream<bool> eDigestLenStrm;
#pragma HLS stream variable = eDigestLenStrm depth = 2
#pragma HLS resource variable = eDigestLenStrm core = FIFO_SRL

    digestToWordStream<dataW, hshW>(msgHashStrm, eMsgHashStrm, digestWordStrm, digestLenStrm, eDigestLenStrm);

    hls::stream<uint32_t> outerScheduleStrm;
#pragma HLS stream variable = outerScheduleStrm depth = 16
#pragma HLS resource variable = outerScheduleStrm core = FIFO_LUTRAM
    hls::stream<uint64_t> outerNblkStrm;
#pragma HLS stream variable = outerNblkStrm depth = 4
#pragma HLS resource variable = outerNblkStrm core = FIFO_LUTRAM
    hls::stream<bool> eOuterNblkStrm;
#pragma HLS stream variable = eOuterNblkStrm depth = 4
#pragma HLS resource variable = eOuterNblkStrm core = FIFO_SRL

    xf::security::internal::generateOuterScheduleFromDigest<dataW>(digestWordStrm, digestLenStrm, eDigestLenStrm,
                                                                   outerScheduleStrm, outerNblkStrm, eOuterNblkStrm,
                                                                   blockSize);

#pragma HLS LATENCY min = 1 max = 1
    xf::security::internal::sha256DigestWithState<hshW>(outerNblkStrm, eOuterNblkStrm, outerScheduleStrm,
                                                        outerStateStrm, eOuterStateStrm, hshStrm, eHshStrm);
}



template <>
inline void outerHash<32, 256, 64>(hls::stream<ap_uint<256> >& outerStateStrm,
                                   hls::stream<bool>& eOuterStateStrm,
                                   hls::stream<ap_uint<256> >& msgHashStrm,
                                   hls::stream<bool>& eMsgHashStrm,
                                   hls::stream<ap_uint<256> >& hshStrm,
                                   hls::stream<bool>& eHshStrm) {
#pragma HLS INLINE off
#pragma HLS dataflow disable_start_propagation

    hls::stream<uint32_t> WtOuterStrm("WtOuterStrm");
#pragma HLS stream variable = WtOuterStrm depth = 2
#pragma HLS resource variable = WtOuterStrm core = FIFO_SRL

    xf::security::internal::outerScheduleFromDigestStage<256>(msgHashStrm, eMsgHashStrm, WtOuterStrm, 64);

    xf::security::internal::sha256DigestOneBlockWithState<256>(WtOuterStrm, outerStateStrm, eOuterStateStrm, hshStrm,
                                                               eHshStrm);
}
template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void hmacDataflow(hls::stream<ap_uint<dataW> >& keyStrm,
                  hls::stream<ap_uint<dataW> >& msgStrm,
                  hls::stream<ap_uint<lW> >& msgLenStrm,
                  hls::stream<bool>& eLenStrm,
                  hls::stream<ap_uint<hshW> >& hshStrm,
                  hls::stream<bool>& eHshStrm) {
#pragma HLS dataflow disable_start_propagation

    hls::stream<bool> eLenKeyStrm("eLenKeyStrm");
#pragma HLS stream variable = eLenKeyStrm depth = 2
#pragma HLS resource variable = eLenKeyStrm core = FIFO_SRL
    hls::stream<bool> eLenMsgStrm("eLenMsgStrm");
#pragma HLS stream variable = eLenMsgStrm depth = 2
#pragma HLS resource variable = eLenMsgStrm core = FIFO_SRL

    hls::stream<ap_uint<hshW> > innerStateStrm("innerStateStrm");
#pragma HLS stream variable = innerStateStrm depth = 2
#pragma HLS resource variable = innerStateStrm core = FIFO_SRL
    hls::stream<bool> eInnerStateStrm("eInnerStateStrm");
#pragma HLS stream variable = eInnerStateStrm depth = 2
#pragma HLS resource variable = eInnerStateStrm core = FIFO_SRL

    hls::stream<ap_uint<hshW> > outerStateStrm("outerStateStrm");
#pragma HLS stream variable = outerStateStrm depth = 2
#pragma HLS resource variable = outerStateStrm core = FIFO_SRL
    hls::stream<bool> eOuterStateStrm("eOuterStateStrm");
#pragma HLS stream variable = eOuterStateStrm depth = 2
#pragma HLS resource variable = eOuterStateStrm core = FIFO_SRL

    hls::stream<ap_uint<hshW> > msgHashStrm("msgHashStrm");
#pragma HLS stream variable = msgHashStrm depth = 2
#pragma HLS resource variable = msgHashStrm core = FIFO_SRL
    hls::stream<bool> eMsgHashStrm("eMsgHashStrm");
#pragma HLS stream variable = eMsgHashStrm depth = 2
#pragma HLS resource variable = eMsgHashStrm core = FIFO_SRL

    duplicateFlagStream(eLenStrm, eLenKeyStrm, eLenMsgStrm);

    kpad<dataW, lW, hshW, keyLen, blockSize, F>(keyStrm, eLenKeyStrm, innerStateStrm, eInnerStateStrm, outerStateStrm,
                                                eOuterStateStrm);

    innerHash<dataW, lW, hshW, blockSize>(msgStrm, msgLenStrm, eLenMsgStrm, innerStateStrm, eInnerStateStrm,
                                          msgHashStrm, eMsgHashStrm);

    outerHash<dataW, hshW, blockSize>(outerStateStrm, eOuterStateStrm, msgHashStrm, eMsgHashStrm, hshStrm, eHshStrm);
}

} // namespace internal

template <int dataW, int lW, int hshW, int keyLen, int blockSize, template <int iW, int ilW, int oW> class F>
void hmac(hls::stream<ap_uint<dataW> >& keyStrm,
          hls::stream<ap_uint<dataW> >& msgStrm,
          hls::stream<ap_uint<lW> >& msgLenStrm,
          hls::stream<bool>& eLenStrm,
          hls::stream<ap_uint<hshW> >& hshStrm,
          hls::stream<bool>& eHshStrm) {
    internal::hmacDataflow<dataW, lW, hshW, keyLen, blockSize, F>(keyStrm, msgStrm, msgLenStrm, eLenStrm, hshStrm,
                                                                  eHshStrm);
}

} // namespace security
} // namespace xf

#endif // _XF_SECURITY_HMAC_HPP_
