
#module load cmake/3.18.5
#module load clang/12.0
mkdir build
#C=clang CXX=clang++ 
cmake -S ./ -B ./build
cmake --build ./build --config Release
cp ./build/run ./run
