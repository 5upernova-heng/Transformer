rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/hengyu/libtorch -DCMAKE_PREFIX_PATH=/opt/OpenBLAS-0.3.26 -DCMAKE_C_COMPILER=/usr/local/bin/gcc -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ .. 
cmake --build . --config Debug --target all -j 18 --
