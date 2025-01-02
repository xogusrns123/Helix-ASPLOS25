//
// Created by meiyixuan on 2024/4/15.
//

#ifndef ZMQ_COMM_UTILS_H
#define ZMQ_COMM_UTILS_H

#include <iostream>
#include <string>
#include <cassert>

void Assert(bool cond, const std::string& err_msg) {
    if (!cond) {
        std::cerr << err_msg << std::endl;
        assert(false);
    }
}

long get_time() {
    // results in micro-seconds
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();
    return microseconds;
}

void log(const std::string& logger, const std::string& msg) {
    std::cout << "[" << logger << "] " << msg << "\n";
}

void custom_free(void* data, void* hint) {
    // No operation; memory managed elsewhere
    // used to avoid zmq copy in message creation
}

uint64_t pack_overhead(uint64_t hop_overheads, uint8_t hop_index, uint16_t overhead_us) {
    // each overhead is 16 bits
    // shift = hop_index * 16
    int shift_amount = hop_index * 16;
    uint64_t mask = ((uint64_t)overhead_us & 0xFFFFULL) << shift_amount;
    return hop_overheads | mask;
}

std::vector<uint16_t> decode_overheads(uint64_t hop_overheads, uint8_t hop_index) {
    std::vector<uint16_t> results;
    results.reserve(hop_index);

    for (int i = 0; i < hop_index; i++) {
        int shift_amount = i * 16;
        // mask out the 16 bits
        uint16_t overhead = (hop_overheads >> shift_amount) & 0xFFFFULL;
        results.push_back(overhead);
    }
    return results;
}

#endif //ZMQ_COMM_UTILS_H
