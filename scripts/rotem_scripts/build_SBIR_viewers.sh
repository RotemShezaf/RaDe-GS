#!/bin/bash

# Activate conda environment if not already activated
if [ -z "$CONDA_PREFIX" ]; then
    echo "Activating conda environment..."
    source /home/rotem.shezaf/miniconda3/etc/profile.d/conda.sh
    conda activate radegs_reproduce_backup
fi

cd /home/rotem.shezaf/RaDe-GS/SIBR_viewers 
rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake Makefile build/

# Set up environment
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# Verify pkg-config can find GTK
echo "Testing pkg-config for GTK..."
pkg-config --modversion gtk+-3.0 || echo "Warning: pkg-config cannot find gtk+-3.0"
pkg-config --cflags gtk+-3.0 || echo "Warning: pkg-config cannot get cflags"

cmake -Bbuild . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DOPENGL_INCLUDE_DIR=$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include \
  -DOPENGL_opengl_LIBRARY=$CONDA_PREFIX/lib/libOpenGL.so \
  -DOPENGL_glx_LIBRARY=$CONDA_PREFIX/lib/libGLX.so

cmake --build build -j24 --target install
