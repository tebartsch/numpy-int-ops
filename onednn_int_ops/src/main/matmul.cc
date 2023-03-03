#include "dnnl.hpp"
#include "matmul.h"

void int8_matmul(int8_t* A, int8_t* B, int32_t* C, int64_t D1, int64_t D2, int64_t M, int64_t K, int64_t N) {
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    // Create a MatMul primitive descriptor for the following op: C_u32 = A_u8 * B_u8
    const int64_t d_D1 = DNNL_RUNTIME_DIM_VAL;
    const int64_t d_D2 = DNNL_RUNTIME_DIM_VAL;
    const int64_t d_M = DNNL_RUNTIME_DIM_VAL;
    const int64_t d_K = DNNL_RUNTIME_DIM_VAL;
    const int64_t d_N = DNNL_RUNTIME_DIM_VAL;
    dnnl::memory::desc a_md({d_D1, d_D2, d_M, d_K}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::abcd); 
    dnnl::memory::desc b_md({d_D1, d_D2, d_K, d_N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::abcd);
    dnnl::memory::desc c_md({d_D1, d_D2, d_M, d_N}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::abcd);
    dnnl::matmul::primitive_desc matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md); 

    // Create Matmul Primitive
    dnnl::matmul matmul_p(matmul_pd);

    // Create memory objects at the locations given by A, B and C
    dnnl::memory A_s8_mem({{D1, D2, M, K}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::abcd}, eng, A);
    dnnl::memory B_s8_mem({{D1, D2, K, N}, dnnl::memory::data_type::s8, dnnl::memory::format_tag::abcd}, eng, B);
    dnnl::memory C_s32_mem({{D1, D2, M, N}, dnnl::memory::data_type::s32, dnnl::memory::format_tag::abcd}, eng, C);

    // Execute the MatMul primitive
    dnnl::stream s(eng);
    matmul_p.execute(s,
        {{DNNL_ARG_SRC, A_s8_mem},
         {DNNL_ARG_WEIGHTS, B_s8_mem},
         {DNNL_ARG_DST, C_s32_mem}});
    s.wait();
}
