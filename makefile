all: a.out

a.out: main.c makefile
	gcc -O3 main.c -lX11 -lGL

run: a.out 
	./a.out