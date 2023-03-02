def get_pointer_to_np_arr(arr, ctype, ffi):
    return ffi.cast(ctype, arr.ctypes.data)
