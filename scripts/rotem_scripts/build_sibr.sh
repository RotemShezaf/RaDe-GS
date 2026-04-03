#!/bin/bash
# Build SIBR viewers - run after setup_sibr.sh

set -e  # Exit on error

cd /home/rotem.shezaf/RaDe-GS/SIBR_viewers

echo "=== Cleaning previous build ==="
rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake Makefile build/

echo "=== Configuring CMake ==="
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/eigen3:$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include/eigen3:$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$C_INCLUDE_PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

cmake -Bbuild . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include/eigen3 -I$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include" \
  -DCMAKE_C_FLAGS="-I$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include" \
  -DCMAKE_BUILD_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_INSTALL_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE

echo "=== Building (this may take a while) ==="
cmake --build build -j24 --target install

echo "=== Build complete! ==="
echo "Binaries installed to: $PWD/install/bin/"
