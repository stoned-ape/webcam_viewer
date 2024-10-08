#pragma once
_Pragma("GCC diagnostic ignored \"-Wmultichar\"");
_Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"");
//X11
#include <X11/Xlib.h>
#include <X11/Xutil.h>
//OpenGL
#include <GL/gl.h>
#include <GL/glx.h>
//C-STD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <sys/timeb.h>
#include <stdarg.h>
//UNIX-STD
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <dirent.h>
//LINUX
// /usr/include/linux/videodev2.h
// /usr/include/linux/v4l2-controls.h
#include <linux/videodev2.h>
#include <gpiod.h>



#define SYSCALL_NOEXIT(call)({ \
    auto syscall_ret=call; \
    if(syscall_ret==(typeof(call))-1){ \
        fprintf(stderr,"syscall error: %d (%s) in function %s at line %d of file %s\n", \
            errno,strerror(errno),__func__,__LINE__,__FILE__); \
        fprintf(stderr,"-> SYSCALL(%s)\n",#call); \
    } \
    syscall_ret; \
})

//exits on error
#define SYSCALL(call)({ \
    auto syscall_ret=SYSCALL_NOEXIT(call); \
    if(syscall_ret==(typeof(call))-1) exit(errno); \
    syscall_ret; \
})

#define CUDA(call){ \
    cudaError_t err=(call); \
    if(err!=0){ \
        fprintf(stderr,"%d -> CUDA(%s) error(%s) in function %s in file %s \n", \
            __LINE__,#call,cudaGetErrorString(err),__func__,__FILE__); \
        exit(1); \
    } \
}


#define PRINT(val,type) printf("%s:\t" type "\n",#val,val);


static void gl_check(const char *file,int line){
    int glerr=glGetError();
    if(glerr!=GL_NO_ERROR){
        const char *ename=NULL;
        #define CASE(x) case x:{ename=#x;break;}
        switch(glerr){
        CASE(GL_NO_ERROR);
        CASE(GL_INVALID_ENUM);
        CASE(GL_INVALID_VALUE);
        CASE(GL_INVALID_OPERATION);
        CASE(GL_STACK_OVERFLOW);
        CASE(GL_STACK_UNDERFLOW);
        CASE(GL_OUT_OF_MEMORY);
        }
        #undef CASE
        fprintf(stderr,"GL_CHECK() failed: error 0x%x (%s) on line %d of file %s\n",glerr,ename,line,file);
        // exit(1);
    }
}

#define GL_CHECK() gl_check(__FILE__,__LINE__)



static double itime(){
    struct timeb now;
    ftime(&now);
    return (double)(now.time%(60*60*24))+now.millitm/1e3;
}

static void print_fps(){
    static double timer;
    double delta=itime()-timer;
    timer+=delta;
    printf("\rfps = %f ",1/delta);
    fflush(stdout);
}

static float map(float t,float t0,float t1,float s0,float s1){
    return s0+(s1-s0)*(t-t0)/(t1-t0);
}

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define clamp(x,a,b) min(max(x,a),b)


#define XDrawString(display,window,gc,x,y,text) XDrawString(display,window,gc,x,y,text,strlen(text))

static int x_draw_printf(  Display *display,Window window,GC gc,int x,int y,
                    const char *format, ...){ 
    static char buf[256];
    va_list ap;
    va_start(ap,format);
    int ret=vsnprintf(buf,sizeof buf,format,ap);
    va_end(ap);
    XDrawString(display,window,gc,x,y,buf);
    return ret;
}
