.PHONY: build run brun clean 

build:	
	nvcc laplace2d.cu -arch=sm_70 -o run_cuda -O3 -std=c++11
run:
	 ./run_cuda

brun: build run

clean: 
		rm ./run_cuda
