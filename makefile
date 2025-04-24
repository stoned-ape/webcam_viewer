all: gyro a.out msi_control 

ARGS=-w -O3 -g

zoomer.o: zoomer.cu zoomer.h common.h makefile
	nvcc $(ARGS) -c zoomer.cu -o zoomer.o

kernels.o: kernels.cu kernels.h common.h makefile
	nvcc $(ARGS) -c kernels.cu -o kernels.o

camera.o: camera.cu camera.h common.h makefile
	nvcc $(ARGS) -c camera.cu -o camera.o

cv_debayer.o: cv_debayer.cu cv_debayer.h common.h makefile
	nvcc $(ARGS) -I/usr/include/opencv4 -c cv_debayer.cu -o cv_debayer.o

usb_api.o: usb_api.cu usb_api.h common.h makefile
	nvcc $(ARGS) -c usb_api.cu -o usb_api.o 

main.o: main.cu cv_debayer.h camera.h kernels.h zoomer.h common.h makefile
	nvcc $(ARGS) -c main.cu -o main.o 


a.out: cv_debayer.o main.o camera.o kernels.o zoomer.o makefile
	nvcc $(ARGS) main.o camera.o kernels.o zoomer.o cv_debayer.o \
		-lX11 -lGL -lgpiod -lpthread \
		`pkg-config --cflags --libs opencv4`
# 	nvlink -o a.out main.o kernels.o zoomer.o -lX11 -lGL mkdi

msi_control.o: msi_control.cu usb_api.h cv_debayer.h camera.h kernels.h zoomer.h common.h makefile
	nvcc $(ARGS) -I/usr/include/opencv4 -c msi_control.cu -o msi_control.o 

MCOBJS= msi_control.cu usb_api.o camera.o kernels.cu zoomer.o cv_debayer.o
MCOBJS2=msi_control.o  usb_api.o camera.o kernels.o  zoomer.o cv_debayer.o
MCLIBS= -lX11 -lGL -lgpiod -lpthread \
		`pkg-config --libs opencv4` \
		`pkg-config --libs libusb-1.0`
MCINCS=	-I/usr/include/opencv4 \
		-I/usr/include/libusb-1.0


msi_control: $(MCOBJS2)  makefile
	nvcc $(MCOBJS2) -o msi_control $(MCLIBS)

msi_control.so: $(MCOBJS) kernels.h makefile
	nvcc --shared -w $(MCOBJS) $(MCINCS) -o msi_control.so $(MCLIBS) --compiler-options -fPIC

# nvlink --arch sm_80 $(MCOBJS) -o msi_control.so $(MCLIBS)
# clang++ --shared -x cuda --cuda-gpu-arch=sm_80 $(MCOBJS) -o msi_control.so $(MCLIBS) -L/usr/local/cuda/lib64 -lcudart 

quat.o: quat.cu quat.h common.h makefile
	nvcc $(ARGS) -c quat.cu -o quat.o

gyro: gyro.cu quat.o msi_control.o common.h makefile
	nvcc gyro.cu quat.o msi_control.o -o gyro -lX11 -lGL `pkg-config --cflags --libs libusb-1.0`

run: a.out 
	./a.out

debug: a.out 
	gdb ./a.out