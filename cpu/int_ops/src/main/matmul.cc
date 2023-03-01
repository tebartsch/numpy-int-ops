#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <chrono>
#include "dnnl.hpp"

using namespace dnnl;
namespace {

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (!handle) throw std::runtime_error("handle is nullptr.");

    uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
    if (!dst) throw std::runtime_error("get_data_handle returned nullptr.");
    for (size_t i = 0; i < size; ++i)
        dst[i] = ((uint8_t *)handle)[i];
    return;
}

void init_vector(std::vector<float> &v) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(0, 1);
    for (auto &e : v)
        e = u(gen);
}

void init_vector(std::vector<int8_t> &v) {
    std::mt19937 gen;
    std::uniform_int_distribution<unsigned int> u(0, 255);
    for (auto &e : v)
        e = static_cast<uint8_t>(u(gen));
}

} // namespace

int number_of_runs = 1;
// Create a MatMul primitive descriptor for the following op:
// C_u32 = A_u8 * B_u8
matmul::primitive_desc matmul_pd_create(const engine &eng) {
    const int64_t D1 = DNNL_RUNTIME_DIM_VAL;
    const int64_t D2 = DNNL_RUNTIME_DIM_VAL;
    const int64_t M = DNNL_RUNTIME_DIM_VAL;
    const int64_t K = DNNL_RUNTIME_DIM_VAL;
    const int64_t N = DNNL_RUNTIME_DIM_VAL;
    memory::desc a_md({D1, D2, M, K}, memory::data_type::s8, {K, 1, 1, 1}); // M x K layout
    memory::desc b_md({D1, D2, K, N}, memory::data_type::s8, {N, 1, 1, 1}); // K x N layout
    memory::desc c_md({D1, D2, M, N}, memory::data_type::s32, {N, 1, 1, 1}); // M x N layout
    // Create a MatMul primitive descriptor
    return matmul::primitive_desc(eng, a_md, b_md, c_md); 
}
void set_random_values(memory &s8_mem) {
    int64_t size = s8_mem.get_desc().get_size() / sizeof(int8_t);
    std::vector<int8_t> s8(size);
    init_vector(s8);
    write_to_dnnl_memory(s8.data(), s8_mem);
}

void infer(const matmul &matmul_p,
           const memory &A_s8_mem, const memory &B_s8_mem, const memory &C_s32_mem,
           const engine &eng) {
    stream s(eng);
    for (int run = 0; run < number_of_runs; ++run)
        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_s8_mem},
                 {DNNL_ARG_WEIGHTS, B_s8_mem},
                 {DNNL_ARG_DST, C_s32_mem}});
    s.wait();
}

void print_mem(memory mem) {
    if (mem.get_desc().get_data_type() == memory::data_type::s8) {
        int8_t *data = (int8_t *)mem.get_data_handle();
        for (int i = 0; i < mem.get_desc().get_size() / sizeof(int8_t); i++)
            std::cout << static_cast<int16_t>(data[i]) << " ";
        std::cout << std::endl;
    } else if (mem.get_desc().get_data_type() == memory::data_type::s32) {
        int32_t *data = (int32_t *)mem.get_data_handle();
        for (int i = 0; i < mem.get_desc().get_size() / sizeof(int32_t); i++)
            std::cout << data[i] << " ";
        std::cout << std::endl;
    } else {
        throw std::runtime_error("Unknown data type");
    }
}

void inference_int8_matmul() {
    engine eng(engine::kind::cpu, 0);

    const int64_t D1 = 16;
    const int64_t D2 = 16;

    const int64_t M = 225;
    const int64_t K = 225;
    const int64_t N = 225;
    auto matmul_pd = matmul_pd_create(eng);
    // Original weights stored as float in a known format
    std::vector<float> B_f32(K * N);
    init_vector(B_f32);
    // Pre-packed weights stored as int8_t
    matmul matmul_p(matmul_pd);
    // Inputs
    memory A_s8_mem({{D1, D2, M, K}, memory::data_type::s8, memory::format_tag::abcd}, eng);
    set_random_values(A_s8_mem);

    memory B_s8_mem({{D1, D2, K, N}, memory::data_type::s8, memory::format_tag::abcd}, eng);
    set_random_values(B_s8_mem);
    // output - no initialization required
    memory C_s32_mem({{D1, D2, M, N}, memory::data_type::s32, memory::format_tag::abcd}, eng);

    auto start_time = std::chrono::high_resolution_clock::now();
    infer(matmul_p, A_s8_mem, B_s8_mem, C_s32_mem, eng);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    std::cout << "Runtime: " << time/std::chrono::milliseconds(1) / 1000.0 << std::endl;

}

int main(int argc, char **argv) {
    inference_int8_matmul();
    return 0;
}