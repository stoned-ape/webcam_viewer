all: a.out

a.out: main.c makefile
	gcc main.c -lX11 -lGL

run: a.out 
	./a.out