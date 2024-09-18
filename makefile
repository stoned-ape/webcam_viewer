all: a.out

ARGS=-w -O3

zoomer.o: zoomer.cu zoomer.h common.h makefile
	nvcc $(ARGS) -c zoomer.cu -o zoomer.o

kernels.o: kernels.cu kernels.h common.h makefile
	nvcc $(ARGS) -c kernels.cu -o kernels.o

camera.o: camera.cu camera.h common.h makefile
	nvcc $(ARGS) -c camera.cu -o camera.o

main.o: main.cu kernels.h zoomer.h common.h makefile
	nvcc $(ARGS) -c main.cu -o main.o 


a.out: main.o camera.o kernels.o zoomer.o makefile
	nvcc $(ARGS) main.o camera.o kernels.o zoomer.o -lX11 -lGL 
# 	nvlink -o a.out main.o kernels.o zoomer.o -lX11 -lGL 

run: a.out 
	./a.out