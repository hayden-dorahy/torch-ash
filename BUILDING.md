# Building torch-ash on CUDA 12.x / GCC 14 (ml1)

This documents the patches required to build torch-ash on a modern stack.
Tested on:

- RHEL 8.10
- CUDA 12.8 at `/usr/local/cuda-12.8`
- GCC 14 via `/opt/rh/gcc-toolset-14`
- PyTorch 2.10+cu128
- Python 3.12
- `uv` package manager (substitute `pip` if preferred)

---

## Step 1 - Clone with submodules

```bash
git clone --recursive git@github.com:hayden-dorahy/torch-ash.git
cd torch-ash
```

If already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Step 2 - Update stdgpu to HEAD

The vendored stdgpu submodule (tag 1.3.0) is not compatible with CUDA 12.x.
Update to the upstream HEAD which adds CUDA 13.x/12.x support:

```bash
cd ext/stdgpu
git fetch
git checkout origin/HEAD
cd ../..
```

## Step 3 - Patch `ext/stdgpu/cmake/Findthrust.cmake`

CUDA 12.8's `thrust/version.h` defines `THRUST_VERSION` with an inline comment
on the same line:

```c
#define THRUST_VERSION 200700 // macro expansion with ## requires this to be a single value
```

The original CMake regex strips the `#define` prefix but leaves the comment,
causing `math()` to fail with a parse error. Change line 15 from:

```cmake
string(REGEX REPLACE "#define THRUST_VERSION[ \t]+" "" THRUST_VERSION_STRING ${THRUST_VERSION_STRING})
```

To:

```cmake
string(REGEX REPLACE "#define THRUST_VERSION[ \t]+([0-9]+).*" "\\1" THRUST_VERSION_STRING ${THRUST_VERSION_STRING})
```

## Step 4 - Patch `setup.py`

Two changes are needed:

**4a - Disable stdgpu benchmarks.** The benchmark suite fails to compile
against Thrust 2.x. Add `-DSTDGPU_BUILD_BENCHMARKS=OFF` to the cmake flags:

```python
cmake_flags = [
    f"-DCMAKE_INSTALL_PREFIX={stdgpu_install_dir}",
    "-DSTDGPU_BUILD_SHARED_LIBS=OFF",
    "-DSTDGPU_BUILD_EXAMPLES=OFF",
    "-DSTDGPU_BUILD_TESTS=OFF",
    "-DSTDGPU_BUILD_BENCHMARKS=OFF",   # add this line
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
]
```

**4b - Add `lib64` to linker search path.** stdgpu installs its static library
to `lib64` on RHEL/CentOS. The original `library_dirs` only lists `lib`:

```python
# Before
library_dirs = [osp.join(stdgpu_install_dir, "lib")]

# After
library_dirs = [osp.join(stdgpu_install_dir, "lib"), osp.join(stdgpu_install_dir, "lib64")]
```

## Step 5 - Patch `ash/src/hashmap_gpu.cuh`

The newer stdgpu API changed the `unordered_map` value type from `thrust::pair`
to `stdgpu::pair<const Key, Value>`. Update the two extractor structs (~line 362):

```cpp
// Before
template <typename Key>
struct IndexExtractor {
    __host__ __device__ int64_t
    operator()(const thrust::pair<Key, int>& x) const {
        return int64_t(x.second);
    }
};

template <typename Key>
struct KeyExtractor {
    __host__ __device__ Key operator()(const thrust::pair<Key, int>& x) const {
        return x.first;
    }
};
```

```cpp
// After
template <typename Key>
struct IndexExtractor {
    __host__ __device__ int64_t
    operator()(const stdgpu::pair<const Key, int>& x) const {
        return int64_t(x.second);
    }
};

template <typename Key>
struct KeyExtractor {
    __host__ __device__ Key operator()(const stdgpu::pair<const Key, int>& x) const {
        return x.first;
    }
};
```

## Step 6 - Install

On ml1, GCC 14 is available via the Red Hat gcc-toolset. Set the environment
to use it alongside CUDA 12.8, then install with `--no-build-isolation` so the
pre-installed PyTorch is visible to the build system:

```bash
export PATH=/opt/rh/gcc-toolset-14/root/usr/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/opt/rh/gcc-toolset-14/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++

uv pip install . --no-build-isolation
```

The build takes around 5-6 minutes.

## Step 7 - Install missing runtime dependency

`ash` imports `einops` at module load time but does not declare it as a
dependency in `setup.py`:

```bash
uv pip install einops
```

## Verification

```python
import ash
from ash import ASHEngine
print("ok")
```
