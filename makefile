all: a.out

a.out: main.cu makefile
	nvcc -O3 main.cu -lX11 -lGL

run: a.out 
	./a.out