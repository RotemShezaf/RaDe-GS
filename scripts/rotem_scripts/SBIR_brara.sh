conda install -c conda-forge     eigen     boost     assimp     glew     glfw     opencv     embree     ffmpeg     libglvnd         xorg-libx11     xorg-libxrandr     xorg-libxinerama     xorg-libxcursor     xorg-libxxf86vm
conda install conda-forge::mesa-libegl-conda-x86_64
conda install -c conda-forge mesa-libgl-devel-conda-x86_64
conda install -c conda-forge \
  glew \
  assimp \
  boost \
  gtk3 \
  opencv \
  glfw \
  ffmpeg \
  eigen = 3.3 \
  embree=3

conda install -c conda-forge libgomp 

conda install -c conda-forge glib 
  # Install all SIBR dependencies in one command
conda install -c conda-forge \
  eigen \
  boost \
  assimp \
  glew \
  glfw \
  opencv \
  embree=3 \
  ffmpeg \
  libglvnd \
  gtk3 \
  libgomp \
  mesa-libegl-cos7-x86_64 \
  mesa-libgl-devel-cos7-x86_64 \
  xorg-libx11 \
  xorg-libxrandr \
  xorg-libxinerama \
  xorg-libxcursor \
  xorg-libxxf86vm

  export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include/eigen3:$C_INCLUDE_PATH
  pkg-config --modversion gtk+-3.0

  cmake -Bbuild . \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_BUILD_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_INSTALL_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
  -DCMAKE_ERROR_DEPRECATED=OFF \
  --no-warn-unused-cli 2>&1 | grep -v "Cannot generate a safe runtime search path"

  cmake -Bbuild . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_BUILD_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_INSTALL_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE \
  -DCMAKE_ERROR_DEPRECATED=OFF \
  --no-warn-unused-cli

  cmake --build . -j24 --target install

  ./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <path to trained model>
/home/rotem.shezaf/RaDe-GS/SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/geussians/Barn

cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX

cd /home/rotem.shezaf/RaDe-GS/SIBR_viewers
rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake Makefile build/
cmake -Bbuild . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_BUILD_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_INSTALL_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE
cmake --build build -j24 --target install

cd /home/rotem.shezaf/RaDe-GS/SIBR_viewers
rm -rf CMakeCache.txt CMakeFiles/ cmake_install.cmake Makefile build/
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/eigen3:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=$CONDA_PREFIX/include/eigen3:$C_INCLUDE_PATH
cmake -Bbuild . \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_CXX_FLAGS="-I$CONDA_PREFIX/include/eigen3" \
  -DCMAKE_BUILD_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_INSTALL_RPATH="/usr/lib/gcc/x86_64-linux-gnu/11:$CONDA_PREFIX/lib" \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE
cmake --build build -j24 --target install