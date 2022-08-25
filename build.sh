#！ /usr/bin/bash

# if [-d "./build"];then
#     cd build
# else
#     mkdir build && cd build
# fi
cd build
echo $(pwd)
rm -rf *
cmake .. -DCMAKE_CUDA_COMPILER=$(which nvcc)
make -j8
