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

#include <libusb-1.0/libusb.h>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_CXX17
// #define GLM_FORCE_AVX2
#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp> 

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

const float pi=3.141592653589793;

static float deg(float rad){return rad*180/pi;}

typedef glm::vec2 _float2;
typedef glm::vec3 _float3;
typedef glm::vec4 _float4;
typedef glm::ivec2 _int2;
typedef glm::ivec3 _int3;
typedef glm::ivec4 _int4;
typedef glm::mat2 _float2x2;
typedef glm::mat3 _float3x3;
typedef glm::mat4 _float4x4;

typedef glm::mat2x3 _float2x3;
typedef glm::mat2x4 _float2x4;

template<typename T> static int print(T arg);
template<typename T,typename ...TV>
int print(T arg,TV ...vargs){
    int x=print<T>(arg);
    return x+print(vargs...);
}
#define PRINT_BASIC_RAW(type,format) \
    template<> int print<type>(type arg){return printf(format,arg);}
#define PRINT_BASIC(type,format) \
    PRINT_BASIC_RAW(type,format) \
    PRINT_BASIC_RAW(type*,"%p ") \
    PRINT_BASIC_RAW(const type*,"%p ")

PRINT_BASIC_RAW(char,"%c");
PRINT_BASIC_RAW(char*,"%s");
PRINT_BASIC_RAW(void*,"%p ");
PRINT_BASIC_RAW(const char*,"%s");
PRINT_BASIC_RAW(const void*,"%p ");
PRINT_BASIC(bool,"%d ");
PRINT_BASIC(short,"%d ");
PRINT_BASIC(int,"%d ");
PRINT_BASIC(long,"%ld ");
PRINT_BASIC(long long,"%lld ");
PRINT_BASIC(unsigned char,"%u ");
PRINT_BASIC(unsigned short,"%u ");
PRINT_BASIC(unsigned int,"%u ");
PRINT_BASIC(unsigned long,"%lu ");
PRINT_BASIC(unsigned long long,"%llu ");
PRINT_BASIC(float,"%f ");
PRINT_BASIC(double,"%f ");
PRINT_BASIC(long double,"%Lf ");

// template<> int print<const wchar_t*>(const wchar_t* arg){return wprintf(L"%s",arg);}
// template<> int print<wchar_t*>(wchar_t* arg){return wprintf(L"%s",arg);}

#undef PRINT_BASIC
#undef PRINT_BASIC_RAW
static int println(){return print('\n');}
template<typename ...TV>
static int println(TV ...vargs){return print(vargs...,'\n');}
#define lprintln(...) println(#__VA_ARGS__,":\t",__VA_ARGS__)


template<> int print<_float2>(_float2 v){return print('[',v.x,',',v.y,']');}
template<> int print<_float3>(_float3 v){return print('[',v.x,',',v.y,',',v.z,']');}
template<> int print<_float4>(_float4 v){return print('[',v.x,',',v.y,',',v.z,',',v.w,']');}

template<> int print<int2>(int2 v){return print('[',v.x,',',v.y,']');}
template<> int print<int3>(int3 v){return print('[',v.x,',',v.y,',',v.z,']');}
template<> int print<int4>(int4 v){return print('[',v.x,',',v.y,',',v.z,',',v.w,']');}

template<> int print<_float2x2>(_float2x2 m){
    _float2 *cols=(_float2*)&m;
    int x=0;
    x+=println('[',cols[0],',');
    x+=println(' ',cols[1],']');
    return x;
}

template<> int print<_float3x3>(_float3x3 m){
    _float3 *cols=(_float3*)&m;
    int x=0;
    x+=println('[',cols[0],',');
    x+=println(' ',cols[1],',');
    x+=println(' ',cols[2],']');
    return x;
}

// static_assert(sizeof(_float3)==4*4);

template<> int print<_float4x4>(_float4x4 m){
    _float4 *cols=(_float4*)&m;
    int x=0;
    x+=println('[',cols[0],',');
    x+=println(' ',cols[1],',');
    x+=println(' ',cols[2],']');
    x+=println(' ',cols[3],']');
    return x;
}

template<uint32_t n>
static void glVertex(glm::vec<n,float,glm::defaultp> v);

template<> void glVertex<2>(_float2 v){glVertex2fv((float*)&v);}
template<> void glVertex<3>(_float3 v){glVertex3fv((float*)&v);}
template<> void glVertex<4>(_float4 v){glVertex4fv((float*)&v);}

inline static void glColor(_float3 c){glColor3f(c.r,c.g,c.b);}
inline static void glTranslate(_float3 c){glTranslatef(c.r,c.g,c.b);}
inline static void glScale(_float3 c){glScalef(c.r,c.g,c.b);}

inline static float pos(bool p){return p?1.0f:-1.0f;}

inline static _float3 m4v3(_float4x4 mtx,_float3 v,bool translate){return (mtx*_float4(v.x,v.y,v.z,(float)translate)).xyz();}
