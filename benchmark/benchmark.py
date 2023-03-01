from time import time
import io

import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import onnx
import onnxruntime as ort


def plot(title: str, filename: str, x: np.ndarray, ys: list[np.ndarray], xlabel: str, labels=list[str]):
    fig, ax = plt.subplots()
    ax.set_title(title)
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.set_yscale("log")
    ax.set_ylabel("time (s)")
    ax.set_xlabel(xlabel)
    ax.legend(loc="upper right")
    fig.savefig(filename)


def time_numpy_matmul(matA: np.ndarray, matB: np.ndarray, repetitions: int = 1):
    stime = time()
    for _ in range(repetitions):
        np.matmul(matA, matB)
    return (time() - stime) / repetitions


def time_torch_int8_matmul(matA: np.ndarray, matB: np.ndarray, repetitions: int = 1):
    out = torch.zeros(matA.shape[:-2] + (matA.shape[-2], matB.shape[-1]),
                      dtype=torch.int32)
    torch_matA = torch.tensor(matA, dtype=torch.int32)
    torch_matB = torch.tensor(matB, dtype=torch.int32)
    stime = time()
    for _ in range(repetitions):
        torch.matmul(torch_matA, torch_matB, out=out)
    return (time() - stime) / repetitions

def time_tensorflow_int8_matmul(matA: np.ndarray, matB: np.ndarray, repetitions: int = 1):
    tf_matA = tf.convert_to_tensor(matA, dtype=tf.int8)
    tf_matB = tf.convert_to_tensor(matB, dtype=tf.int8)
    stime = time()
    for _ in range(repetitions):
        result = tf.matmul(tf_matA, tf_matB, output_type=tf.int32)
    return (time() - stime) / repetitions

def time_onnx_int8_matmul(matA: np.ndarray, matB: np.ndarray, repetitions: int = 1):
    input_a_name, input_b_name, output_name = "input_a", "input_b", "output"
    input_a = onnx.helper.make_tensor_value_info(input_a_name, onnx.TensorProto.INT8, shape=None)
    input_b = onnx.helper.make_tensor_value_info(input_b_name, onnx.TensorProto.INT8, shape=None)
    output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.INT32, shape=None)
    node = onnx.helper.make_node(
        name="MatMulInteger",
        op_type="MatMulInteger",
        inputs=[input_a_name, input_b_name],
        outputs=[output_name],
    )
    graph_def = onnx.helper.make_graph(
        nodes=[node],
        name="MatMul",
        inputs=[input_a, input_b],
        outputs=[output],
    )
    onnx_model = onnx.helper.make_model(graph_def, producer_name="numpy-quant-test")
    onnx_model.opset_import[0].version = 13
    onnx_bytes = io.BytesIO()
    onnx.save_model(onnx_model, onnx_bytes)
    session = ort.InferenceSession(onnx_bytes.getvalue())

    stime = time()
    for _ in range(repetitions):
        result = session.run([output_name], {
            input_a_name: matA,
            input_b_name: matB,
        })

    return (time() - stime) / repetitions


def run_benchmark(rng):
    max_size = 225
    step = 25
    N = np.arange(25, max_size + 1, step)
    np_float32_time = np.zeros(N.shape)
    np_int8_time = np.zeros(N.shape)
    torch_int8_time = np.zeros(N.shape)
    tensorflow_int8_time = np.zeros(N.shape)
    onnxruntime_int8_time = np.zeros(N.shape)

    k, l = 16, 12

    for i, n in tqdm(list(enumerate(N))):
        float32_matA = rng.normal(size=(k, l, n, n)).astype(np.float32)
        float32_matB = rng.normal(size=(k, l, n, n)).astype(np.float32)
        int8_matA = rng.integers(-2**7, 2**7-1, size=(k, l, n, n), dtype=np.int8)
        int8_matB = rng.integers(-2**7, 2**7-1, size=(k, l, n, n), dtype=np.int8)

        # repetitions = max_size - n + 1
        repetitions = 3

        np_float32_time[i] = time_numpy_matmul(float32_matA, float32_matB, repetitions)

        np_int8_time[i] = time_numpy_matmul(int8_matA, int8_matB, repetitions)
        torch_int8_time[i] = time_torch_int8_matmul(int8_matA, int8_matB, repetitions)
        tensorflow_int8_time[i] = time_tensorflow_int8_matmul(int8_matA, int8_matB, repetitions)
        onnxruntime_int8_time[i] = time_onnx_int8_matmul(int8_matA, int8_matB, repetitions)

    plot("int8 matrix-multiplication on CPU\nnormalized to np-float32",
         "result_normalized.png",
         N, 
         [
            np_float32_time / np_float32_time, 
            np_int8_time / np_float32_time, 
            torch_int8_time / np_float32_time,
            tensorflow_int8_time / np_float32_time,
            onnxruntime_int8_time / np_float32_time,
         ],
         labels=['np (float32)','np (int8)', 'torch (int8)', 'tensorflow (int8)', 'onnxruntime (int8)'],
         xlabel=f"matrix size: {k}x{l}xNxN")
    
    plot("int8 matrix-multiplication on CPU",
         "result.png",
         N, 
         [
            np_float32_time, 
            np_int8_time, 
            torch_int8_time,
            tensorflow_int8_time,
            onnxruntime_int8_time,
         ],
         labels=['np (float32)','np (int8)', 'torch (int8)', 'tensorflow (int8)', 'onnxruntime (int8)'],
         xlabel=f"matrix size: {k}x{l}xNxN")


if __name__ == '__main__':
    rng = np.random.default_rng()
    run_benchmark(rng)
