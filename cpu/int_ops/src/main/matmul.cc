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

#include "matmul.h"

using namespace dnnl;

// Create a MatMul primitive descriptor for the following op:
// C_u32 = A_u8 * B_u8
matmul::primitive_desc matmul_pd_create(const engine &eng) {
    const int64_t D1 = DNNL_RUNTIME_DIM_VAL;
    const int64_t D2 = DNNL_RUNTIME_DIM_VAL;
    const int64_t M = DNNL_RUNTIME_DIM_VAL;
    const int64_t K = DNNL_RUNTIME_DIM_VAL;
    const int64_t N = DNNL_RUNTIME_DIM_VAL;
    memory::desc a_md({D1, D2, M, K}, memory::data_type::s8, memory::format_tag::abcd); 
    memory::desc b_md({D1, D2, K, N}, memory::data_type::s8, memory::format_tag::abcd);
    memory::desc c_md({D1, D2, M, N}, memory::data_type::s32, memory::format_tag::abcd);
    // Create a MatMul primitive descriptor
    return matmul::primitive_desc(eng, a_md, b_md, c_md); 
}

void matmul_execute(const matmul &matmul_p,
                    const memory &A_s8_mem, const memory &B_s8_mem, const memory &C_s32_mem,
                    const engine &eng) {
    stream s(eng);
    matmul_p.execute(s,
        {{DNNL_ARG_SRC, A_s8_mem},
         {DNNL_ARG_WEIGHTS, B_s8_mem},
         {DNNL_ARG_DST, C_s32_mem}});
    s.wait();
}

void int8_matmul(int8_t* A, int8_t* B, int32_t* C, int64_t D1, int64_t D2, int64_t M, int64_t K, int64_t N) {
    engine eng(engine::kind::cpu, 0);
    auto matmul_pd = matmul_pd_create(eng);

    matmul matmul_p(matmul_pd);
    memory A_s8_mem({{D1, D2, M, K}, memory::data_type::s8, memory::format_tag::abcd}, eng, A);
    memory B_s8_mem({{D1, D2, K, N}, memory::data_type::s8, memory::format_tag::abcd}, eng, B);
    memory C_s32_mem({{D1, D2, M, N}, memory::data_type::s32, memory::format_tag::abcd}, eng, C);

    matmul_execute(matmul_p, A_s8_mem, B_s8_mem, C_s32_mem, eng);
}
