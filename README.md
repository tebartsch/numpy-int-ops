# python-integer-matmul

## Benchmark of Alternatives

```bash
cd benchmark-alternatives
python3.10 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## CPU

### Installation

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
./IntOps
```