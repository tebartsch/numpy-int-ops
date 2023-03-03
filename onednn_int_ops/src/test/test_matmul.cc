#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include "dnnl.hpp"

#include "matmul.h"

using namespace dnnl;

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

void set_random_values(int8_t* arr, int64_t n) {
    std::vector<int8_t> vec(n);
    init_vector(vec);
    for (int i = 0; i < n; i++)
        arr[i] = static_cast<int8_t>(vec[i]);
}

void print_int8_arr(int8_t* arr, int64_t n) {
    for (int i = 0; i < n; i++)
        std::cout << static_cast<int16_t>(arr[i]) << " ";
    std::cout << std::endl;
}

void print_int32_arr(int32_t* arr, int64_t n) {
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    int8_t A_arr[] = {1, 2, 3, 4};
    int8_t* A = A_arr;
    int8_t B_arr[] = {2, 2, 2, 2};
    int8_t* B = B_arr;
    int32_t C_arr[] = {0, 0, 0, 0};
    int32_t* C = C_arr;
    int8_matmul(A_arr, B_arr, C_arr, 1, 1, 2, 2, 2);

    print_int8_arr(A, 4);
    print_int8_arr(B, 4);
    print_int32_arr(C, 4);
    
    return 0;
}