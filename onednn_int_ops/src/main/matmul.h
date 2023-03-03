#ifndef MATMUL_H
#define MATMUL_H

#include <stdint.h>

int main(int argc, char **argv);

extern "C" {
    void int8_matmul(int8_t* A, int8_t* B, int32_t* C, int64_t D1, int64_t D2, int64_t M, int64_t K, int64_t N);
} //end extern "C"

#endif // MATMUL_H