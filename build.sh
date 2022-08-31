#！ /usr/bin/bash

if [ ! -d "build"];then
    mkdir build
else
    echo "we already have build"
fi

cd build
echo $(pwd)
rm -rf *
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc)
make -j8
