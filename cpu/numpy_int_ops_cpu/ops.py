from cffi import FFI
import numpy as np

import numpy_int_ops_cpu
from numpy_int_ops_cpu.helpers import get_pointer_to_np_arr

# Instantiate FFI
dllURI = f"{numpy_int_ops_cpu.__path__[0]}/libIntOps.so"
funDef = """
void int8_matmul(int8_t* A, int8_t* B, int32_t* C, int64_t D1, int64_t D2, int64_t M, int64_t K, int64_t N);
"""
ffi = FFI()
cpp_lib = ffi.dlopen(dllURI)
ffi.cdef(funDef)

def int8_matmul(A: np.ndarray[None, np.int8], B: np.ndarray[None, np.int8]) -> np.ndarray[None, np.int32]:
    assert A.dtype == np.int8, "A must be int8"
    assert B.dtype == np.int8, "B must be int8"
    assert len(A.shape) == 4, "A must be 4D tensor"
    assert len(B.shape) == 4, "B must be 4D tensor"
    assert A.shape[:-4] == B.shape[:-4], f"Cannot matmul A (shape={A.shape}) and B (shape={B.shape})"
    assert A.shape[:-3] == B.shape[:-3], f"Cannot matmul A (shape={A.shape}) and B (shape={B.shape})"
    assert A.shape[-1] == B.shape[-2], f"Cannot matmul A (shape={A.shape}) and B (shape={B.shape})"
    D1 = A.shape[-4]
    D2 = A.shape[-3]
    K, M, N = A.shape[-2], A.shape[-1], B.shape[-1]
    C = np.empty(shape=(D1, D2, K, N), dtype=np.int32)
    A_pointer = get_pointer_to_np_arr(A, "int8_t*", ffi)
    B_pointer = get_pointer_to_np_arr(B, "int8_t*", ffi)
    C_pointer = get_pointer_to_np_arr(C, "int32_t*", ffi)
    cpp_lib.int8_matmul(A_pointer, B_pointer, C_pointer, D1, D2, K, M, N)
    return C
