.PHONY: build run brun clean 

build:	
		cmake -E make_directory ./build
		C=mpic CXX=mpic++ cmake -S ./ -B ./build
		cmake --build ./build --config Release

run:
	numactl --cpunodebind=1 mpirun -np 9 ./build/run

brun: build run

clean: 
		rm -r ./build
