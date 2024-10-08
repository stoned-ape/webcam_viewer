all: a.out

ARGS=-w -O3 -g

zoomer.o: zoomer.cu zoomer.h common.h makefile
	nvcc $(ARGS) -c zoomer.cu -o zoomer.o

kernels.o: kernels.cu kernels.h common.h makefile
	nvcc $(ARGS) -c kernels.cu -o kernels.o

camera.o: camera.cu camera.h common.h makefile
	nvcc $(ARGS) -c camera.cu -o camera.o

cv_debayer.o: cv_debayer.cu cv_debayer.h common.h makefile
	nvcc $(ARGS) -I/usr/include/opencv4 -c cv_debayer.cu -o cv_debayer.o

main.o: main.cu cv_debayer.h camera.h kernels.h zoomer.h common.h makefile
	nvcc $(ARGS) -c main.cu -o main.o 


a.out: cv_debayer.o main.o camera.o kernels.o zoomer.o makefile
	nvcc $(ARGS) main.o camera.o kernels.o zoomer.o cv_debayer.o \
		-lX11 -lGL -lgpiod -lpthread \
		`pkg-config --cflags --libs opencv4`
# 	nvlink -o a.out main.o kernels.o zoomer.o -lX11 -lGL 

run: a.out 
	./a.out

debug: a.out 
	gdb ./a.out