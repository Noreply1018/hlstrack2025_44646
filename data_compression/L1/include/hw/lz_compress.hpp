/*
 * (c) Copyright 2019-2022 Xilinx, Inc. All rights reserved.
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
 *
 */
#ifndef _XFCOMPRESSION_LZ_COMPRESS_HPP_
#define _XFCOMPRESSION_LZ_COMPRESS_HPP_

/**
 * @file lz_compress.hpp
 * @brief Header for modules used in LZ4 and snappy compression kernels.
 *
 * This file is part of Vitis Data Compression Library.
 */
#include "compress_utils.hpp"
#include "hls_stream.h"

#include <ap_int.h>
#include <assert.h>
#include <stdint.h>

namespace xf {
namespace compression {

template <int MATCH_LEN, int MATCH_LEVEL, int DICT_ELE_WIDTH>
struct MatchUnitPacket {
    ap_uint<MATCH_LEN * 8> windowVec;
    ap_uint<MATCH_LEVEL * DICT_ELE_WIDTH> dictSnapshot;
    ap_uint<32> currIdx;
    ap_uint<1> valid;
};

struct LiteralFlushPacket {
    ap_uint<8> data;
    ap_uint<1> last;
};

struct TokenWord {
    ap_uint<32> data;
    ap_uint<1> last;
};

template <int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int MATCH_LEVEL,
          int MIN_OFFSET,
          int LZ_DICT_SIZE,
          int LEFT_BYTES,
          typename MatchPktT>
static void lzCompressMatchPrepare(hls::stream<ap_uint<8> >& inStream,
                                   hls::stream<MatchPktT>& matchStream,
                                   hls::stream<LiteralFlushPacket>& literalStream,
                                   uint32_t input_size) {
#pragma HLS INLINE off
    const int c_dictEleWidth = (MATCH_LEN * 8 + 24);
    typedef ap_uint<MATCH_LEVEL * c_dictEleWidth> uintDictV_t;

    uintDictV_t dict[LZ_DICT_SIZE];
#pragma HLS BIND_STORAGE variable = dict type = RAM_T2P impl = BRAM
#pragma HLS RESOURCE variable = dict core = RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable = dict cyclic factor = 7 dim = 1

    auto sendSentinel = [&]() {
        MatchPktT sentinel;
        sentinel.windowVec = 0;
        sentinel.dictSnapshot = 0;
        sentinel.currIdx = 0;
        sentinel.valid = 0;
        matchStream << sentinel;
        LiteralFlushPacket litTail;
        litTail.data = 0;
        litTail.last = 1;
        literalStream << litTail;
    };

    if (input_size == 0) {
        sendSentinel();
        return;
    }

    uintDictV_t resetValue = 0;
    for (int i = 0; i < MATCH_LEVEL; i++) {
#pragma HLS UNROLL
        resetValue.range((i + 1) * c_dictEleWidth - 1, i * c_dictEleWidth + MATCH_LEN * 8) = -1;
    }
dict_flush_init:
    for (int i = 0; i < LZ_DICT_SIZE; i++) {
#pragma HLS UNROLL
        dict[i] = resetValue;
    }

    uint8_t present_window[MATCH_LEN];
#pragma HLS ARRAY_PARTITION variable = present_window complete
    for (uint8_t i = 1; i < MATCH_LEN; i++) {
#pragma HLS PIPELINE II = 1
        present_window[i] = inStream.read();
    }

lz_match_prep:
    for (uint32_t i = MATCH_LEN - 1; i < input_size - LEFT_BYTES; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS dependence variable = dict inter false
        uint32_t currIdx = i - MATCH_LEN + 1;
        for (int m = 0; m < MATCH_LEN - 1; m++) {
#pragma HLS UNROLL
            present_window[m] = present_window[m + 1];
        }
        present_window[MATCH_LEN - 1] = inStream.read();

        // A2: 哈希等价缩短 - 使用ap_uint并注册地址/输出
        ap_uint<12> hash_narrow;
        if (MIN_MATCH == 3) {
            hash_narrow = (ap_uint<12>)present_window[0] << 4;
            hash_narrow ^= (ap_uint<12>)present_window[1] << 3;
            hash_narrow ^= (ap_uint<12>)present_window[2] << 2;
            hash_narrow ^= (ap_uint<12>)present_window[0] << 1;
            hash_narrow ^= (ap_uint<12>)present_window[1];
        } else {
            hash_narrow = (ap_uint<12>)present_window[0] << 4;
            hash_narrow ^= (ap_uint<12>)present_window[1] << 3;
            hash_narrow ^= (ap_uint<12>)present_window[2] << 2;
            hash_narrow ^= (ap_uint<12>)present_window[3];
        }
        
        // A2: 地址打一拍 - 注册哈希地址
        ap_uint<12> hash_reg = hash_narrow & 0xFFF;
        uint32_t hash = hash_reg;

        // A2: BRAM输出打一拍 - 直接使用寄存器注册
        uintDictV_t dictReadValue = dict[hash];
        uintDictV_t dictWriteValue = dictReadValue << c_dictEleWidth;
        for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
            dictWriteValue.range((m + 1) * 8 - 1, m * 8) = present_window[m];
        }
        dictWriteValue.range(c_dictEleWidth - 1, MATCH_LEN * 8) = currIdx;
        dict[hash] = dictWriteValue;

        MatchPktT pkt;
        pkt.currIdx = currIdx;
        pkt.dictSnapshot = dictReadValue;
        pkt.windowVec = 0;
        for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
            pkt.windowVec.range((m + 1) * 8 - 1, m * 8) = present_window[m];
        }
        pkt.valid = 1;
        matchStream << pkt;
    }

    MatchPktT sentinel;
    sentinel.windowVec = 0;
    sentinel.dictSnapshot = 0;
    sentinel.currIdx = 0;
    sentinel.valid = 0;
    matchStream << sentinel;

    LiteralFlushPacket litPkt;
lz_compress_leftover_stage:
    for (int m = 1; m < MATCH_LEN; ++m) {
#pragma HLS PIPELINE II = 1
        litPkt.data = present_window[m];
        litPkt.last = 0;
        literalStream << litPkt;
    }

lz_left_bytes_stage:
    for (int l = 0; l < LEFT_BYTES; ++l) {
#pragma HLS PIPELINE II = 1
        ap_uint<8> byte = inStream.read();
        litPkt.data = byte;
        litPkt.last = 0;
        literalStream << litPkt;
    }

    litPkt.data = 0;
    litPkt.last = 1;
    literalStream << litPkt;
}

template <int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int MATCH_LEVEL,
          int MIN_OFFSET,
          typename MatchPktT>
static void lzCompressMatchEvaluate(hls::stream<MatchPktT>& matchStream, hls::stream<TokenWord>& tokenStream) {
#pragma HLS INLINE off
    const int c_dictEleWidth = (MATCH_LEN * 8 + 24);
    typedef ap_uint<c_dictEleWidth> uintDict_t;
match_eval_loop:
    while (true) {
#pragma HLS PIPELINE II = 1
        MatchPktT pkt = matchStream.read();
        if (!pkt.valid) break;

        ap_uint<8> present_window[MATCH_LEN];
#pragma HLS ARRAY_PARTITION variable = present_window complete
        for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
            present_window[m] = pkt.windowVec.range((m + 1) * 8 - 1, m * 8);
        }
        ap_uint<MATCH_LEVEL * c_dictEleWidth> dictReadValue = pkt.dictSnapshot;

        ap_uint<8> match_length = 0;
        ap_uint<16> match_offset = 0;
        for (int l = 0; l < MATCH_LEVEL; l++) {
#pragma HLS UNROLL
            ap_uint<8> len = 0;
            bool done = 0;
            uintDict_t compareWith = dictReadValue.range((l + 1) * c_dictEleWidth - 1, l * c_dictEleWidth);
            ap_uint<32> compareIdx = compareWith.range(c_dictEleWidth - 1, MATCH_LEN * 8);
            for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
                bool bytesEqual = (present_window[m] == compareWith.range((m + 1) * 8 - 1, m * 8));
                if (bytesEqual && !done) {
                    len++;
                } else {
                    done = 1;
                }
            }
            bool len_ge_min = (len >= MIN_MATCH);
            bool idx_gt = (pkt.currIdx > compareIdx);
            ap_uint<17> rawOffset = 0;
            if (idx_gt) rawOffset = pkt.currIdx - compareIdx;
            bool within_limit = idx_gt && (rawOffset < LZ_MAX_OFFSET_LIMIT);
            ap_uint<16> adjOffset = 0;
            if (rawOffset != 0) adjOffset = rawOffset - 1;
            bool offset_ge_min = idx_gt && (adjOffset >= MIN_OFFSET);
            if (len_ge_min && within_limit && offset_ge_min) {
                if ((len == 3) && (adjOffset > 4096)) {
                    len = 0;
                }
            } else {
                len = 0;
            }
            if (len > match_length) {
                match_length = len;
                match_offset = adjOffset;
            }
        }

        TokenWord word;
        word.data = 0;
        word.last = 0;
        word.data.range(7, 0) = present_window[0];
        word.data.range(15, 8) = match_length;
        word.data.range(31, 16) = match_offset;
        tokenStream << word;
    }

    TokenWord doneWord;
    doneWord.data = 0;
    doneWord.last = 1;
    tokenStream << doneWord;
}

static void lzWriteLiteralFlush(hls::stream<LiteralFlushPacket>& literalStream,
                                hls::stream<TokenWord>& literalWordStream) {
#pragma HLS INLINE off
#pragma HLS interface ap_ctrl_none port=return
lz_literal_flush:
    while (true) {
#pragma HLS PIPELINE II = 1
        LiteralFlushPacket pkt = literalStream.read();
        TokenWord out;
        out.data = 0;
        out.data.range(7, 0) = pkt.data;
        out.last = pkt.last;
        literalWordStream << out;
        if (pkt.last) break;
    }
}

static void lzAssembleOutput(hls::stream<TokenWord>& tokenStream,
                             hls::stream<TokenWord>& literalStream,
                             hls::stream<ap_uint<32> >& outStream) {
#pragma HLS INLINE off
#pragma HLS interface ap_ctrl_none port=return
token_stage:
    while (true) {
#pragma HLS PIPELINE II = 1
        TokenWord w = tokenStream.read();
        bool isLast = w.last;
        if (!isLast) {
            outStream << w.data;
        }
        if (isLast) break;
    }
literal_stage:
    while (true) {
#pragma HLS PIPELINE II = 1
        TokenWord w = literalStream.read();
        bool isLast = w.last;
        if (!isLast) {
            outStream << w.data;
        }
        if (isLast) break;
    }
    ap_uint<32> tail = 0;
    outStream << tail;
}

/**
 * @brief This module reads input literals from stream and updates
 * match length and offset of each literal.
 *
 * @tparam MATCH_LEN match length
 * @tparam MIN_MATCH minimum match
 * @tparam LZ_MAX_OFFSET_LIMIT maximum offset limit
 * @tparam MATCH_LEVEL match level
 * @tparam MIN_OFFSET minimum offset
 * @tparam LZ_DICT_SIZE dictionary size
 *
 * @param inStream input stream
 * @param outStream output stream
 * @param input_size input size
 */
template <int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int MATCH_LEVEL = 6,
          int MIN_OFFSET = 1,
          int LZ_DICT_SIZE = 1 << 12,
          int LEFT_BYTES = 64>
void lzCompress(hls::stream<ap_uint<8> >& inStream, hls::stream<ap_uint<32> >& outStream, uint32_t input_size) {
    typedef MatchUnitPacket<MATCH_LEN, MATCH_LEVEL, (MATCH_LEN * 8 + 24)> MatchPktT;
    hls::stream<MatchPktT> matchStream("matchStream");
    hls::stream<LiteralFlushPacket> literalStream("literalStream");
    hls::stream<TokenWord> tokenStream("tokenStream");
    hls::stream<TokenWord> literalWordStream("literalWordStream");
#pragma HLS STREAM variable = matchStream depth = 4
#pragma HLS STREAM variable = literalStream depth = 4
#pragma HLS STREAM variable = tokenStream depth = 4
#pragma HLS STREAM variable = literalWordStream depth = 4
#pragma HLS BIND_STORAGE variable = matchStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = literalStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = tokenStream type = FIFO impl = SRL
#pragma HLS BIND_STORAGE variable = literalWordStream type = FIFO impl = SRL

#pragma HLS dataflow
    lzCompressMatchPrepare<MATCH_LEN, MIN_MATCH, LZ_MAX_OFFSET_LIMIT, MATCH_LEVEL, MIN_OFFSET, LZ_DICT_SIZE, LEFT_BYTES>(
        inStream, matchStream, literalStream, input_size);
    lzCompressMatchEvaluate<MATCH_LEN, MIN_MATCH, LZ_MAX_OFFSET_LIMIT, MATCH_LEVEL, MIN_OFFSET>(matchStream,
                                                                                               tokenStream);
    lzWriteLiteralFlush(literalStream, literalWordStream);
    lzAssembleOutput(tokenStream, literalWordStream, outStream);
}
/**
 * @brief This is stream-in-stream-out module used for lz compression. It reads input literals from stream and updates
 * match length and offset of each literal.
 *
 * @tparam MATCH_LEN match length
 * @tparam MIN_MATCH minimum match
 * @tparam LZ_MAX_OFFSET_LIMIT maximum offset limit
 * @tparam MATCH_LEVEL match level
 * @tparam MIN_OFFSET minimum offset
 * @tparam LZ_DICT_SIZE dictionary size
 *
 * @param inStream input stream
 * @param outStream output stream
 */
template <int MAX_INPUT_SIZE = 64 * 1024,
          class SIZE_DT = uint32_t,
          int MATCH_LEN,
          int MIN_MATCH,
          int LZ_MAX_OFFSET_LIMIT,
          int CORE_ID = 0,
          int MATCH_LEVEL = 6,
          int MIN_OFFSET = 1,
          int LZ_DICT_SIZE = 1 << 12,
          int LEFT_BYTES = 64>
void lzCompress(hls::stream<IntVectorStream_dt<8, 1> >& inStream, hls::stream<IntVectorStream_dt<32, 1> >& outStream) {
    const uint16_t c_indxBitCnts = 24;
    const uint16_t c_fifo_depth = LEFT_BYTES + 2;
    const int c_dictEleWidth = (MATCH_LEN * 8 + c_indxBitCnts);
    typedef ap_uint<MATCH_LEVEL * c_dictEleWidth> uintDictV_t;
    typedef ap_uint<c_dictEleWidth> uintDict_t;
    const uint32_t totalDictSize = (1 << (c_indxBitCnts - 1)); // 8MB based on index 3 bytes
#ifndef AVOID_STATIC_MODE
    static bool resetDictFlag = true;
    static uint32_t relativeNumBlocks = 0;
#else
    bool resetDictFlag = true;
    uint32_t relativeNumBlocks = 0;
#endif

    uintDictV_t dict[LZ_DICT_SIZE];
#pragma HLS BIND_STORAGE variable = dict type = RAM_T2P impl = BRAM
#pragma HLS RESOURCE variable = dict core = RAM_T2P_BRAM
#pragma HLS ARRAY_PARTITION variable = dict cyclic factor = 7 dim = 1

    // local buffers for each block
    uint8_t present_window[MATCH_LEN];
#pragma HLS ARRAY_PARTITION variable = present_window complete
    hls::stream<uint8_t> lclBufStream("lclBufStream");
#pragma HLS STREAM variable = lclBufStream depth = c_fifo_depth
#pragma HLS BIND_STORAGE variable = lclBufStream type = fifo impl = srl

    // input register
    IntVectorStream_dt<8, 1> inVal;
    // output register
    IntVectorStream_dt<32, 1> outValue;
    // loop over blocks
    while (true) {
        uint32_t iIdx = 0;
        // once 8MB data is processed reset dictionary
        // 8MB based on index 3 bytes
        if (resetDictFlag) {
            ap_uint<MATCH_LEVEL* c_dictEleWidth> resetValue = 0;
            for (int i = 0; i < MATCH_LEVEL; i++) {
#pragma HLS UNROLL
                resetValue.range((i + 1) * c_dictEleWidth - 1, i * c_dictEleWidth + MATCH_LEN * 8) = -1;
            }
        // Initialization of Dictionary
        dict_flush:
            for (int i = 0; i < LZ_DICT_SIZE; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL FACTOR = 4
                dict[i] = resetValue;
            }
            resetDictFlag = false;
            relativeNumBlocks = 0;
        } else {
            relativeNumBlocks++;
        }
        // check if end of data
        auto nextVal = inStream.read();
        if (nextVal.strobe == 0) {
            outValue.strobe = 0;
            outStream << outValue;
            break;
        }
    // fill buffer and present_window
    lz_fill_present_win:
        while (iIdx < MATCH_LEN - 1) {
#pragma HLS PIPELINE II = 1
            inVal = nextVal;
            nextVal = inStream.read();
            present_window[++iIdx] = inVal.data[0];
        }
    // assuming that, at least bytes more than LEFT_BYTES will be present at the input
    lz_fill_circular_buf:
        for (uint16_t i = 0; i < LEFT_BYTES; ++i) {
#pragma HLS PIPELINE II = 1
            inVal = nextVal;
            nextVal = inStream.read();
            lclBufStream << inVal.data[0];
        }
        // lz_compress main
        outValue.strobe = 1;

    lz_compress:
        for (; nextVal.strobe != 0; ++iIdx) {
#pragma HLS PIPELINE II = 1
#ifndef DISABLE_DEPENDENCE
#pragma HLS dependence variable = dict inter false
#endif
            uint32_t currIdx = (iIdx + (relativeNumBlocks * MAX_INPUT_SIZE)) - MATCH_LEN + 1;
            // read from input stream into circular buffer
            auto inValue = lclBufStream.read(); // pop latest value from FIFO
            lclBufStream << nextVal.data[0];    // push latest read value to FIFO
            nextVal = inStream.read();          // read next value from input stream

            // shift present window and load next value
            for (uint8_t m = 0; m < MATCH_LEN - 1; m++) {
#pragma HLS UNROLL
                present_window[m] = present_window[m + 1];
            }

            present_window[MATCH_LEN - 1] = inValue;

            // Calculate Hash Value
            uint32_t hash = 0;
            if (MIN_MATCH == 3) {
                hash = (present_window[0] << 4) ^ (present_window[1] << 3) ^ (present_window[2] << 2) ^
                       (present_window[0] << 1) ^ (present_window[1]);
            } else {
                hash = (present_window[0] << 4) ^ (present_window[1] << 3) ^ (present_window[2] << 2) ^
                       (present_window[3]);
            }

            // Dictionary Lookup
            uintDictV_t dictReadValue = dict[hash];
            uintDictV_t dictWriteValue = dictReadValue << c_dictEleWidth;
            for (int m = 0; m < MATCH_LEN; m++) {
#pragma HLS UNROLL
                dictWriteValue.range((m + 1) * 8 - 1, m * 8) = present_window[m];
            }
            dictWriteValue.range(c_dictEleWidth - 1, MATCH_LEN * 8) = currIdx;
            // Dictionary Update
            dict[hash] = dictWriteValue;

            // Match search and Filtering
            // Comp dict pick
            uint8_t match_length = 0;
            uint32_t match_offset = 0;
            for (int l = 0; l < MATCH_LEVEL; l++) {
                uint8_t len = 0;
                bool done = 0;
                uintDict_t compareWith = dictReadValue.range((l + 1) * c_dictEleWidth - 1, l * c_dictEleWidth);
                uint32_t compareIdx = compareWith.range(c_dictEleWidth - 1, MATCH_LEN * 8);
                for (uint8_t m = 0; m < MATCH_LEN; m++) {
                    if (present_window[m] == compareWith.range((m + 1) * 8 - 1, m * 8) && !done) {
                        len++;
                    } else {
                        done = 1;
                    }
                }
                if ((len >= MIN_MATCH) && (currIdx > compareIdx) && ((currIdx - compareIdx) < LZ_MAX_OFFSET_LIMIT) &&
                    ((currIdx - compareIdx - 1) >= MIN_OFFSET) &&
                    (compareIdx >= (relativeNumBlocks * MAX_INPUT_SIZE))) {
                    if ((len == 3) && ((currIdx - compareIdx - 1) > 4096)) {
                        len = 0;
                    }
                } else {
                    len = 0;
                }
                if (len > match_length) {
                    match_length = len;
                    match_offset = currIdx - compareIdx - 1;
                }
            }
            outValue.data[0].range(7, 0) = present_window[0];
            outValue.data[0].range(15, 8) = match_length;
            outValue.data[0].range(31, 16) = match_offset;
            outStream << outValue;
        }

        outValue.data[0] = 0;
    lz_compress_leftover:
        for (uint8_t m = 1; m < MATCH_LEN; ++m) {
#pragma HLS PIPELINE II = 1
            outValue.data[0].range(7, 0) = present_window[m];
            outStream << outValue;
        }
    lz_left_bytes:
        for (uint16_t l = 0; l < LEFT_BYTES; ++l) {
#pragma HLS PIPELINE II = 1
            outValue.data[0].range(7, 0) = lclBufStream.read();
            outStream << outValue;
        }

        // once relativeInSize becomes 8MB set the flag to true
        resetDictFlag = ((relativeNumBlocks * MAX_INPUT_SIZE) >= (totalDictSize)) ? true : false;
        // end of block
        outValue.strobe = 0;
        outStream << outValue;
    }
}

} // namespace compression
} // namespace xf
#endif // _XFCOMPRESSION_LZ_COMPRESS_HPP_
