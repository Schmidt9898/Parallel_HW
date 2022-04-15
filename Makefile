.PHONY: build run brun clean 

build:	
	cmake -E make_directory $(CURDIR)/release
	CXX=nvc++ C=nvc cmake -S $(CURDIR) -B $(CURDIR)/release 
	cmake --build $(CURDIR)/release 
run:
	 ./release/run 128 10000

brun: build run

clean: 
		rm ./release/run
