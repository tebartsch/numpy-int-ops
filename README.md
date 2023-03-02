# python-integer-matmul

Fast integer operations for numpy.

Supported Ops:

 - `int8 x int8 -> int32` matrix multiplication for 4-dimensional arrays

![](benchmark/result.png)

## CPU

### Installation

On Fedora

```bash
cd cpu/int_ops

# Create a ubuntu container with gcc and cmake
podman build -t ubuntu-build .
# Build oneDNN in container
podman run -v "$PWD":"$PWD":z ubuntu-build bash -c "cd $PWD && bash container_build_onednn.sh"
# Install oneDNN to install
podman run -v "$PWD":"$PWD":z ubuntu-build bash -c "cd $PWD && bash container_install_onednn.sh"
```

```bash
cd cpu/int_ops
mkdir build && cd build
cmake ..
make
cp libIntOps.so ../../numpy_int_ops_cpu/.
```

## Benchmark

```bash
cd benchmark
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install ../cpu/numpy_int_ops_cpu
pip install -r requirements.txt
```