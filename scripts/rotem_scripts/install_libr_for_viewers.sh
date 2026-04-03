#!/bin/bash
# Install libraries for SIBR viewers
set -e

# Don't use mesa packages - they require incompatible sysroot
# Don't use opencv/gtk3 - too many conflicts with Python 3.9
echo "=== Installing SIBR dependencies ==="

# Remove ALL boost-related packages including libboost
echo "=== Removing ALL boost packages ==="
conda list | grep -i boost | awk '{print $1}' | xargs -r conda remove -y --force 2>/dev/null || true

# Install dependencies (without boost first)
echo "=== Installing core dependencies ==="
conda install -c conda-forge -y \
  eigen=3.3.9 \
  embree=3 \
  ffmpeg=4.4 \
  glew \
  glfw \
  libglu \
  assimp \
  libgomp \
  gtk3 \
  xorg-libx11 \
  xorg-libxrandr \
  xorg-libxinerama \
  xorg-libxcursor \
  xorg-libxxf86vm


# Install boost 1.82 with Python 3.9 compatibility
echo "=== Installing boost 1.82 ==="
conda install -c conda-forge -y boost=1.82.0 boost-cpp=1.82.0

echo "=== Installing numpy for Python bindings ==="
conda install -c conda-forge -y "numpy<2.0"

echo "=== Manually installing OpenCV 4.5.5 (bypassing conda solver) ==="
cd /tmp
wget -q https://anaconda.org/conda-forge/libopencv/4.5.5/download/linux-64/libopencv-4.5.5-py39ha3d0060_12.tar.bz2 -O libopencv.tar.bz2
wget -q https://anaconda.org/conda-forge/py-opencv/4.5.5/download/linux-64/py-opencv-4.5.5-py39h25bab4e_12.tar.bz2 -O py-opencv.tar.bz2

echo "Extracting OpenCV to $CONDA_PREFIX..."
tar -xjf libopencv.tar.bz2 -C $CONDA_PREFIX
tar -xjf py-opencv.tar.bz2 -C $CONDA_PREFIX

rm -f libopencv.tar.bz2 py-opencv.tar.bz2
cd /home/rotem.shezaf/RaDe-GS
conda install -c conda-forge -y "glib>=2.82"
conda install -c conda-forge -y glib glib-tools libglib
echo "=== Creating symlinks to system OpenGL ==="
ln -sf /usr/lib/x86_64-linux-gnu/libOpenGL.so.0 $CONDA_PREFIX/lib/libOpenGL.so
ln -sf /usr/lib/x86_64-linux-gnu/libGLX.so.0 $CONDA_PREFIX/lib/libGLX.so

echo "=== Setting include paths ==="
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH
echo "=== Done ===" 

