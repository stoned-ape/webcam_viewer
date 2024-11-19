#include "kernels.h"


__global__ void unpack(const void *packed_data,void *unpacked_data,int n){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=n) return;

    struct __attribute__((packed)) packed_t{
        uint16_t a:12,b:12;
    }; 
    static_assert(sizeof(packed_t)==3,"");

    packed_t p=((const packed_t*)packed_data)[idx/2];

    ((uint16_t*)unpacked_data)[idx]=(idx%2==0)?p.a<<4:p.b<<4;

    return;
}

__global__ void bayer_to_rgb(
        const void *bayer_data,void *rgb_data,
        int w,int h){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=2*w*h) return; 

    struct rgb_t{uint16_t r,g,b,a;}; 

    rgb_t *rgb_array=(typeof(rgb_array))rgb_data;
   
    uint16_t *x=(uint16_t*)bayer_data;
    // f();
    int i=idx/w;
    int j=idx%w;

    uint16_t v=x[idx]; //works
    

    rgb_t pixel;
    pixel.a=0xffff;

    //  0 1 = j%2
    //0 B G
    //1 G R

    if      (i%2==0 && j%2==1){//green
        pixel.r=x[(i+1)*w+(j+0)];
        pixel.g=v;
        pixel.b=x[(i+0)*w+(j-1)];
    }else if(i%2==1 && j%2==1){ //red
        pixel.r=v;
        pixel.g=((x[(i-1)*w+(j-0)])+(x[(i-0)*w+(j-1)]))/2;
        pixel.b=  x[(i-1)*w+(j-1)];
    }else if(i%2==0 && j%2==0){ //blue
        pixel.r=  x[(i+1)*w+(j+1)];
        pixel.g=((x[(i+1)*w+(j+0)])+(x[(i+0)*w+(j+1)]))/2;
        pixel.b=v;
    }else if(i%2==1 && j%2==0){ //green
        pixel.r=x[(i+0)*w+(j+1)];
        pixel.g=v;
        pixel.b=x[(i-1)*w+(j+0)];
    }

    // pixel.r=v;
    // pixel.g=v;
    // pixel.b=v;
    
    rgb_array[idx]=pixel;
}



__global__ void yuv_to_rgb(
        const void *yuv_data,void *rgb_data,
        int width,int height,unsigned int fmt){

    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=2*width*height) return; 

    struct{uint8_t a[4];}     *yuv_array=(decltype(yuv_array))yuv_data,yuv_ent;
    struct{uint16_t r,g,b,a;}  *rgb_array=(decltype(rgb_array))rgb_data;
    const int wh=width*height;
    char *fmtc=(char*)&fmt;
    int ui,vi,y0i,y1i;

    // memcpy(rgb_data,yuv_data,width*height*2);
    // return;

    assert(fmt=='YUYV' || fmt=='YVYU' || fmt=='UYVY' || fmt=='VYUY');
    
    for(int i=0;i<4;i++){
        switch(fmtc[i]){
        case 'Y':
            if(i<2) y0i=i;
            else    y1i=i;
            break;
        case 'U':ui=i;break;
        case 'V':vi=i;break;
        }
    }
    

    // for(int idx=0;idx<wh;idx++)
    {
        yuv_ent=yuv_array[idx>>1];
        int u=yuv_ent.a[ui];
        int v=yuv_ent.a[vi];
        int y=((idx&1)==1)?yuv_ent.a[y0i]:yuv_ent.a[y1i];

        int r=(int)(y+1.370705*(v-128));
        int g=(int)(y-0.698001*(v-128)-0.337633*(u-128));
        int b=(int)(y+1.732446*(u-128));

        r=(r<0)?0:(r>255)?255:r;
        g=(g<0)?0:(g>255)?255:g;
        b=(b<0)?0:(b>255)?255:b;

        // return;
        rgb_array[idx].b=b<<8;
        rgb_array[idx].g=g<<8;
        rgb_array[idx].r=r<<8;
    }
}
