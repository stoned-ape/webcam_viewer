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

#define SYSCALL_NOEXIT(call)({ \
    __auto_type syscall_ret=call; \
    if(syscall_ret==(typeof(call))-1){ \
        fprintf(stderr,"syscall error: %d (%s) in function %s at line %d of file %s\n", \
            errno,strerror(errno),__func__,__LINE__,__FILE__); \
        fprintf(stderr,"-> SYSCALL(%s)\n",#call); \
    } \
    syscall_ret; \
})

//exits on error
#define SYSCALL(call)({ \
    __auto_type syscall_ret=SYSCALL_NOEXIT(call); \
    if(syscall_ret==(typeof(call))-1) exit(errno); \
    syscall_ret; \
})

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



double itime(){
    struct timeb now;
    ftime(&now);
    return (double)(now.time%(60*60*24))+now.millitm/1e3;
}

void print_fps(){
    static double timer;
    double delta=itime()-timer;
    timer+=delta;
    printf("\rfps = %f ",1/delta);
    fflush(stdout);
}

void make_bmp(const char *file_name,int width,int height,void *pixels){
    typedef struct __attribute__((packed)){
        uint16_t magic; //0x4d42
        uint32_t file_size;
        uint32_t app; //0
        uint32_t offset; //54
        uint32_t info_size; //40
        int32_t width;
        int32_t height;
        uint16_t planes; //1
        uint16_t bits_per_pix; //32 (four bytes)
        uint32_t comp; //0
        uint32_t comp_size; //size in bytes of the pixel buffer (w*h*4)
        uint32_t xres; //0
        uint32_t yres; //0
        uint32_t cols_used; //0
        uint32_t imp_cols; //0
    }bmp_header;

    static_assert(sizeof(bmp_header)==54,"");

    bmp_header head={0};
    head.magic=0x4d42;
    head.app=0;
    head.offset=sizeof(bmp_header);
    head.info_size=40;
    head.width=width;
    head.height=height;
    head.planes=1;
    head.bits_per_pix=32;
    head.comp_size=width*height*head.bits_per_pix/8;
    head.file_size=sizeof(bmp_header)+head.comp_size;

    uint8_t *ptr=(uint8_t*)pixels;
    for(int i=0;i<width*height;i++){
        uint8_t tmp=ptr[4*i+0];
        ptr[4*i+0]=ptr[4*i+2];
        ptr[4*i+2]=tmp;
    }

    int bmp=SYSCALL(open(file_name,O_CREAT|O_RDWR,0777));
    unsigned long long m;
    m=SYSCALL(write(bmp,&head,sizeof head));
    assert(m==sizeof(bmp_header));
    m=SYSCALL(write(bmp,pixels,head.comp_size));
    assert(m==head.comp_size);
    SYSCALL(close(bmp));
}

typedef __attribute__((vector_size(8))) float float2;

float2 make_float2(float x,float y){
    float2 f={x,y};
    return f;
}

float map(float t,float t0,float t1,float s0,float s1){
    return s0+(s1-s0)*(t-t0)/(t1-t0);
}


float2 map2(float2 t,float2 t0,float2 t1,float2 s0,float2 s1){
    return s0+(s1-s0)*(t-t0)/(t1-t0);
}

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define clamp(x,a,b) min(max(x,a),b)

typedef struct{
    float2 corner;
    float2 size;
}rect;

typedef struct{
    float2 image_size;
    float2 screen_size;
    rect image_rect;
    rect screen_rect;
    float new_scale;
    float old_scale;
    float max_scale;
    float min_scale;
    bool panning;
    float2 old_center;
}zoomer_t;

void zoomer_init(zoomer_t *self,float2 _image_size,float2 _screen_size){
    self->panning=false;
    self->new_scale=1;
    self->old_scale=1;
    self->image_size=_image_size;
    self->screen_size=_screen_size;
    self->image_rect.corner=make_float2(0,0);
    self->screen_rect.corner=make_float2(0,0);
    self->image_rect.size=_image_size;
    self->screen_rect.size=_screen_size;
    self->max_scale=200;
    self->min_scale=1;
    self->old_center=make_float2(.5f,.5f);
}

void zoomer_do_zoom(zoomer_t *self,const float2 center,const float delta_scale){
    self->new_scale=clamp(self->new_scale*delta_scale,self->min_scale,self->max_scale);

    const float2 screen_tl=self->screen_rect.corner;
    const float2 screen_br=self->screen_rect.corner+self->screen_rect.size;

    float2 image_tl=self->image_rect.corner;
    float2 image_br=self->image_rect.corner+self->image_rect.size;

    const float2 image_center=map2(center,screen_tl,screen_br,image_tl,image_br);

    const float scale_ratio=self->new_scale/self->old_scale;

    image_tl-=image_center;
    image_br-=image_center;

    image_tl/=scale_ratio;
    image_br/=scale_ratio;

    image_tl+=image_center;
    image_br+=image_center;

    self->image_rect.corner=image_tl;
    self->image_rect.size=image_br-image_tl;

    if(self->image_rect.corner[0]<0) self->image_rect.corner[0]=0;
    if(self->image_rect.corner[1]<0) self->image_rect.corner[1]=0;

    if(self->image_rect.corner[0]+self->image_rect.size[0]>self->image_size[0]){ 
        self->image_rect.corner[0]=self->image_size[0]-self->image_rect.size[0];
    }
    if(self->image_rect.corner[1]+self->image_rect.size[1]>self->image_size[1]){ 
        self->image_rect.corner[1]=self->image_size[1]-self->image_rect.size[1];
    }

    self->old_scale=self->new_scale;
    self->old_center=center;
}

void zoomer_begin_pan(zoomer_t *self,const float2 center){
    self->panning=true;
    self->old_center=center;
}

void zoomer_repeat_pan(zoomer_t *self,const float2 center){
    if(!self->panning) return;
    self->image_rect.corner-=(center-self->old_center)/self->new_scale;
    zoomer_do_zoom(self,center,1);
}

void zoomer_end_pan(zoomer_t *self){
    self->panning=false;
}

void glTexCoord(float2 v){
    glTexCoord2fv((float*)&v);
}
void glVertex(float2 v){
    glVertex2fv((float*)&v);
}

void stretch_blt(float2 dst_corner,float2 dst_size,float2 src_corner,float2 src_size){
    glPushMatrix();
    glScalef(1,-1,1);
    glTranslatef(-1,-1,0);
    glScalef(2,2,1);

    glEnable(GL_TEXTURE_2D);
    glColor3f(1,1,1);
    glBegin(GL_TRIANGLE_STRIP);

    glTexCoord(src_corner+src_size*make_float2(0,0));
    glVertex(  dst_corner+dst_size*make_float2(0,0));

    glTexCoord(src_corner+src_size*make_float2(0,1));
    glVertex(  dst_corner+dst_size*make_float2(0,1));

    glTexCoord(src_corner+src_size*make_float2(1,0));
    glVertex(  dst_corner+dst_size*make_float2(1,0));

    glTexCoord(src_corner+src_size*make_float2(1,1));
    glVertex(  dst_corner+dst_size*make_float2(1,1));

    glEnd();
    glDisable(GL_TEXTURE_2D);

    glPopMatrix();
}

 




int start_stream(int fd){
    enum v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    return SYSCALL(ioctl(fd,VIDIOC_STREAMON,&type));
}

int stop_stream(int fd){
    enum v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    return SYSCALL(ioctl(fd,VIDIOC_STREAMOFF,&type));
}


#define XDrawString(display,window,gc,x,y,text) XDrawString(display,window,gc,x,y,text,strlen(text))

int x_draw_printf(  Display *display,Window window,GC gc,int x,int y,
                    const char *format, ...){ 
    static char buf[256];
    va_list ap;
    va_start(ap,format);
    int ret=vsnprintf(buf,sizeof buf,format,ap);
    va_end(ap);
    XDrawString(display,window,gc,x,y,buf);
    return ret;
}

void yuv_to_rgb(const void *yuv_data,void *rgb_data,int width,int height,unsigned int fmt){
    struct{uint8_t a[4];}     *yuv_array=yuv_data,yuv_ent;
    struct{uint8_t r,g,b,a;}  *rgb_array=rgb_data;
    const int wh=width*height;
    char *fmtc=(char*)&fmt;
    int ui,vi,y0i,y1i;

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
    
    for(int idx=0;idx<wh;idx++){
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

        rgb_array[idx].b=b;
        rgb_array[idx].g=g;
        rgb_array[idx].r=r;
    }
}

#define NUM_BUFS 1

typedef struct{
    char dev_name[16];
    int fd;
    int w;
    int h;
    void *buf_ptr;
    int buf_size;
    void *draw_buf;
    uint32_t tex_id;
    bool valid;

    struct v4l2_capability cap;
    struct v4l2_fmtdesc fmtdesc;
    struct v4l2_format fmt;
}cam_data_t;


cam_data_t deinit_cam(cam_data_t *cd){
    SYSCALL(close(cd->fd));
    free(cd->draw_buf);
    cd->draw_buf=NULL;
    SYSCALL(munmap(cd->buf_ptr,cd->buf_size));
    memset(cd,0,sizeof(cam_data_t));
    cd->fd=-1;
}

void draw_cam_info(Display *display,Window info_window,GC gc,cam_data_t *cd){
    XClearWindow(display,info_window);
    XDrawString(display,info_window,gc,5 ,20,"selected device:");
    XDrawString(display,info_window,gc,15,40,cd->dev_name);
    XDrawString(display,info_window,gc,15,60,cd->cap.card);
    XDrawString(display,info_window,gc,15,80,cd->cap.driver);
    XDrawString(display,info_window,gc,15,100,cd->cap.bus_info);
    XDrawString(display,info_window,gc,15,120,cd->fmtdesc.description);
    x_draw_printf(display,info_window,gc,15,140,"%dx%d",cd->w,cd->h);
}

cam_data_t init_cam(const char *dev_name){
    cam_data_t cd={0};
    cd.valid=1;
    strncpy(cd.dev_name,dev_name,sizeof cd.dev_name);

    int fd=SYSCALL(open(dev_name,O_RDWR));
    cd.fd=fd;

    struct v4l2_capability cap={0};
    SYSCALL(ioctl(fd,VIDIOC_QUERYCAP,&cap));

    PRINT(cap.driver,"%s");
    PRINT(cap.bus_info,"%s");
    PRINT(cap.card,"%s");
    PRINT(cap.capabilities,"%x");
    cd.cap=cap;

    // #define num_ranges (8)
    const int ranges[][2]={
        {V4L2_CID_USER_BASE,43},
        {V4L2_CID_MPEG_BASE,644},
        {V4L2_CID_CAMERA_CLASS_BASE,51},
        {V4L2_CID_JPEG_CLASS_BASE,7},
        {V4L2_CID_IMAGE_SOURCE_CLASS_BASE,7},
        {V4L2_CID_IMAGE_PROC_CLASS_BASE,5},
        {V4L2_CID_RF_TUNER_CLASS_BASE,91},
        {V4L2_CID_DETECT_CLASS_BASE,4},
    };
    const int num_ranges=sizeof(ranges)/sizeof(ranges[0]);

    for(int j=0;j<num_ranges;j++){
        struct v4l2_query_ext_ctrl queryctrl={0};
        for(int i=ranges[j][0];i<=ranges[j][0]+ranges[j][1];i++){
            queryctrl.id=i;
            if(-1!=ioctl(fd,VIDIOC_QUERY_EXT_CTRL,&queryctrl)){
                PRINT(queryctrl.name,"%s");
                switch(queryctrl.type){
                    #define CASE(x) \
                        case x: \
                            printf("\t%s:0x%x\n",#x,x);

                    CASE(V4L2_CTRL_TYPE_INTEGER       );
                        break;
                    CASE(V4L2_CTRL_TYPE_BOOLEAN       );
                        break;
                    CASE(V4L2_CTRL_TYPE_MENU          );
                        break;
                    CASE(V4L2_CTRL_TYPE_BUTTON        );
                        break;
                    CASE(V4L2_CTRL_TYPE_INTEGER64     );
                        break;
                    CASE(V4L2_CTRL_TYPE_CTRL_CLASS    );
                        break;
                    CASE(V4L2_CTRL_TYPE_STRING        );
                        break;
                    CASE(V4L2_CTRL_TYPE_BITMASK       );
                        break;
                    CASE(V4L2_CTRL_TYPE_INTEGER_MENU  );
                        break;
                    #undef CASE
                }
                printf("\tid:0x%x\n",queryctrl.id);
                printf("\tmin:%lld\n",queryctrl.minimum);
                printf("\tmax:%lld\n",queryctrl.maximum);
                printf("\tdefault:%lld\n",queryctrl.default_value);

                struct v4l2_control control;
                control.id=queryctrl.id;
                control.value=queryctrl.default_value;
                assert(control.value==queryctrl.default_value);
                ioctl(fd,VIDIOC_S_CTRL,&control);

                if(-1!=SYSCALL_NOEXIT(ioctl(fd,VIDIOC_G_CTRL,&control))){
                    printf("\tcurrent:%d\n",control.value);
                }

                puts("");
            }

        }
    }

    struct v4l2_control control;
    control.id=V4L2_CID_EXPOSURE_AUTO;
    control.value=1;
    SYSCALL(ioctl(fd,VIDIOC_S_CTRL,&control));

    // struct v4l2_control control;
    control.id=0x009a0902;
    control.value=315;
    // SYSCALL(ioctl(fd,VIDIOC_S_CTRL,&control));


    struct v4l2_fmtdesc fmtdesc={0};
    fmtdesc.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmtdesc.index=0;
    int pixel_format=0;
    while(0==ioctl(fd,VIDIOC_ENUM_FMT,&fmtdesc)){
        printf("%s: ",(char*)&fmtdesc.pixelformat);

        const int pf=fmtdesc.pixelformat;
        if(pf=='YUYV' || pf=='YVYU' || pf=='UYVY' || pf=='VYUY'){
            pixel_format=pf;
        }

        struct v4l2_frmsizeenum frmsize={0};
        frmsize.type=V4L2_FRMSIZE_TYPE_DISCRETE;
        frmsize.pixel_format=fmtdesc.pixelformat;
        frmsize.index=0;
        while(0==ioctl(fd,VIDIOC_ENUM_FRAMESIZES,&frmsize)){
            if(frmsize.type==V4L2_FRMSIZE_TYPE_DISCRETE){
                printf("%dx%d",frmsize.discrete.width,frmsize.discrete.height);

                struct v4l2_frmivalenum frmival={0};
                frmival.index=0;
                frmival.pixel_format=frmsize.pixel_format;
                frmival.width =frmsize.discrete.width;
                frmival.height=frmsize.discrete.height;
                while(0==ioctl(fd,VIDIOC_ENUM_FRAMEINTERVALS,&frmival)){
                    if(frmival.type!=V4L2_FRMIVAL_TYPE_DISCRETE){
                        printf(" %0.fHz",1.0*frmival.stepwise.min.denominator/frmival.stepwise.min.numerator);
                    }else{
                        printf(" %0.fHz",1.0*frmival.discrete.denominator/frmival.discrete.numerator);
                    }
                    frmival.index++;    
                }

                printf(", ");

            }
            frmsize.index++;
        }

        
        printf("%s\n",fmtdesc.description);
        fmtdesc.index++;
    }
    cd.fmtdesc=fmtdesc;


    struct v4l2_format fmt={0};
    fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;

    fmt.fmt.pix.width=1280;
    fmt.fmt.pix.height=720;
    // fmt.fmt.pix.width=640;
    // fmt.fmt.pix.height=480;
    fmt.fmt.pix.pixelformat=pixel_format;
    fmt.fmt.pix.field=V4L2_FIELD_NONE;

    int res=SYSCALL_NOEXIT(ioctl(fd,VIDIOC_S_FMT,&fmt));
    if(res==-1){
        cd.valid=0;
        SYSCALL(close(cd.fd));
        return cd;
    }
    
    PRINT(fmt.fmt.pix.width,"%d");
    PRINT(fmt.fmt.pix.height,"%d");
    PRINT(fmt.fmt.pix.field,"%d");
    PRINT((char*)&fmt.fmt.pix.pixelformat,"%s");

    cd.w=fmt.fmt.pix.width;
    cd.h=fmt.fmt.pix.height;

    cd.fmt=fmt;
    

    struct v4l2_requestbuffers reqbuf={0};

    reqbuf.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory=V4L2_MEMORY_MMAP;
    // reqbuf.memory=V4L2_MEMORY_USERPTR;
    reqbuf.count=1;
    SYSCALL(ioctl(fd,VIDIOC_REQBUFS,&reqbuf));
    assert(reqbuf.count==NUM_BUFS);

    PRINT(reqbuf.type,"%d");
    PRINT(reqbuf.memory,"%d");
    PRINT(reqbuf.count,"%d");
    PRINT(reqbuf.capabilities,"%d");

    int bytes_per_pixel=4;

    for(int i=0;i<reqbuf.count;i++){
        struct v4l2_buffer buffer={0};
        buffer.type=reqbuf.type;
        buffer.memory=V4L2_MEMORY_MMAP;
        buffer.index=i;
        SYSCALL(ioctl(fd,VIDIOC_QUERYBUF,&buffer));

        cd.buf_ptr=SYSCALL(mmap(NULL,buffer.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,buffer.m.offset));
        assert(cd.buf_ptr);
        cd.buf_size=buffer.length;

        bytes_per_pixel=buffer.length/fmt.fmt.pix.width/fmt.fmt.pix.height;

        PRINT(buffer.length,"%u");
        PRINT(buffer.bytesused,"%u");
        PRINT(buffer.m.offset,"%u");

    }

    PRINT(bytes_per_pixel,"%d");

    for(int i=0;i<reqbuf.count;i++){
        struct v4l2_buffer buffer={0};
        buffer.type=reqbuf.type;
        buffer.memory=V4L2_MEMORY_MMAP;
        buffer.index=i;
        SYSCALL(ioctl(fd,VIDIOC_QBUF,&buffer));
    }
    
    cd.draw_buf=malloc(4*fmt.fmt.pix.width*fmt.fmt.pix.height);
    assert(cd.draw_buf);

    glGenTextures(1, &cd.tex_id);
    assert(cd.tex_id>0);
    glBindTexture(GL_TEXTURE_2D, cd.tex_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cd.w, cd.h, 0, GL_RGBA, GL_UNSIGNED_BYTE,cd.draw_buf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    return cd;
}


int main(){

    static struct{char name[16];char card[32];} devs_info[10]={0};
    int devs_count=0;
    int devs_select=0;

    #define MAX_DEVS (sizeof(devs_info)/sizeof(devs_info[0]))

    DIR *dev_dir=opendir("/dev");
    assert(dev_dir);
    struct dirent *entry;
    while((entry=readdir(dev_dir))!=NULL){
        if(0==memcmp(entry->d_name,"video",5)){
           if(devs_count>=MAX_DEVS) break;
           memcpy(devs_info[devs_count].name,"/dev/",5);
           strcpy(devs_info[devs_count].name+5,entry->d_name);
           devs_count++;
        }
    }
    for(int i=0;i<devs_count;i++){
        int fd=SYSCALL(open(devs_info[i].name,O_RDWR));
        if(fd>0){
            struct v4l2_capability cap={0};
            SYSCALL(ioctl(fd,VIDIOC_QUERYCAP,&cap));
            memcpy(devs_info[i].card,cap.card,32);
            SYSCALL(close(fd));
        }


        puts(devs_info[i].name);
    }
    closedir(dev_dir);
    assert(devs_count>0);
  

    Display *display=NULL;
    int screen=-1;
    Window window={0};
    GC gc=NULL;


    int win_w=1280;
    int win_h=720;
    int side_bar_w=200;

    display=XOpenDisplay(NULL);
    assert(display);
    screen=DefaultScreen(display);
    window=XCreateSimpleWindow(display,RootWindow(display,screen),0,0,win_w+side_bar_w,win_h,1,
        0,WhitePixel(display,screen));
    XSelectInput(display,window,ExposureMask|KeyPressMask);

    Atom wm_delete_window=XInternAtom(display,"WM_DELETE_WINDOW",False);
    XSetWMProtocols(display, window, &wm_delete_window, 1);



    XMapWindow(display,window);
    XFlush(display);
    PRINT(DefaultDepth(display,screen),"%d");
    gc=DefaultGC(display,screen);
    
    Window stream_button=XCreateSimpleWindow(
        display,
        window,
        10,10,
        side_bar_w-20,40,
        1,
        0x0,0x00aaaaaa
    );
    XMapWindow(display,stream_button);
    XSelectInput(display,stream_button,ExposureMask|ButtonPressMask);

    Window save_button=XCreateSimpleWindow(
        display,
        window,
        10,60,
        side_bar_w-20,40,
        1,
        0x0,0x00aaaaaa
    );
    XMapWindow(display,save_button);
    XSelectInput(display,save_button,ExposureMask|ButtonPressMask);


    Window dev_buttons[MAX_DEVS];
    for(int i=0;i<devs_count;i++){
        dev_buttons[i]=XCreateSimpleWindow(
            display,
            window,
            10,130+i*50,
            side_bar_w-20,40,
            1,
            0x0,0x00aaaaaa
        );
        XMapWindow(display,dev_buttons[i]);
        XSelectInput(display,dev_buttons[i],ExposureMask|ButtonPressMask);
        
    }

    Window info_window=XCreateSimpleWindow(
        display,
        window,
        10,140+devs_count*50,
        side_bar_w-20,200,
        1,
        0x0,0x00eeeeee
    );
    XMapWindow(display,info_window);




    int att[]={GLX_RGBA,GLX_DEPTH_SIZE,24,GLX_DOUBLEBUFFER,None};
    XVisualInfo *vi=glXChooseVisual(display, 0, att);
    assert(vi);

    XSetWindowAttributes swa;
    Colormap cmap=XCreateColormap(display,RootWindow(display,screen),vi->visual,AllocNone);
    swa.colormap=cmap;
    swa.event_mask=ExposureMask|KeyPressMask;
    Window gl_window=XCreateWindow(display,window,side_bar_w,0,win_w,win_h,0,vi->depth,
                        InputOutput,vi->visual,CWColormap|CWEventMask,&swa);
    XMapWindow(display,gl_window);
    XSelectInput(display,gl_window,ButtonPressMask|ButtonReleaseMask);
    

    GLXContext glc=glXCreateContext(display,vi,NULL,GL_TRUE);
    glXMakeCurrent(display,gl_window,glc);

    puts((const char *)glGetString(GL_VERSION));

    cam_data_t cd=init_cam(devs_info[devs_select].name);

    zoomer_t zoomer;
    zoomer_init(&zoomer,make_float2(1,1),make_float2(1,1));


    bool is_streaming=false;
    if(cd.valid){
        start_stream(cd.fd);
        is_streaming=true;
    }

    bool running=true;
    while(running){
        const float aspect=(cd.h/(float)cd.w)*(win_w/(float)win_h);
        int root_x,root_y,win_x,win_y;
        Window child;
        Window root;
        unsigned int mask;
        XQueryPointer(display,gl_window,&root,&child,&root_x,&root_y,&win_x,&win_y,&mask);
        float2 mouse;
        mouse[0]=map(win_x,0,win_w,-1/aspect,1/aspect);
        mouse[1]=map(win_y,0,win_h,1,-1);
        
        const float margin=(1-1/aspect)/2;
        float2 center;
        center[0]=map(win_x,0,win_w,margin,1-margin);
        center[1]=map(win_y,0,win_h,0,1);

        XEvent e={0};
        if(XCheckTypedEvent(display,ClientMessage,&e)){
            if(e.xclient.data.l[0]==wm_delete_window){
                running=false;
                printf("window close event received\n");
                goto app_done;
            }
        }
        if(XCheckTypedEvent(display,Expose,&e)){
            puts("expose");

            XDrawString(display,stream_button,gc,10,20,"stop stream");
            XDrawString(display,save_button,gc,10,20,"save image");
            XDrawString(display,window,gc,10,120,"choose video device:");

            for(int i=0;i<devs_count;i++){
                XDrawString(display,dev_buttons[i],gc,10,15,devs_info[i].name);
                XDrawString(display,dev_buttons[i],gc,10,35,devs_info[i].card);
            }
            draw_cam_info(display,info_window,gc,&cd);
        }
        if(XCheckTypedEvent(display,ButtonPress,&e)){
            struct v4l2_buffer buffer={0};
            buffer.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory=V4L2_MEMORY_MMAP;
            if(e.xbutton.window==stream_button){
                printf("stream button clicked\n");

                if(is_streaming){ 
                    SYSCALL(ioctl(cd.fd,VIDIOC_DQBUF,&buffer));
                    stop_stream(cd.fd);
                    is_streaming=false;

                    XClearWindow(display,stream_button);
                    XDrawString(display,stream_button,gc,10,20,"start stream");

                }else if(cd.valid){ 
                    SYSCALL(ioctl(cd.fd,VIDIOC_QBUF,&buffer));
                    start_stream(cd.fd);
                    is_streaming=true;

                    XClearWindow(display,stream_button);
                    char *button_text="stop stream";
                    XDrawString(display,stream_button,gc,10,20,"stop stream");

                }
            }else if(e.xbutton.window==save_button && cd.valid){
                printf("saving image\n");

                make_bmp("img.bmp",cd.w,cd.h,cd.draw_buf);
            }

            for(int i=0;i<devs_count;i++) if(e.xbutton.window==dev_buttons[i]){
                printf("device %d button clicked\n",i);
                if(is_streaming){ 
                    SYSCALL(ioctl(cd.fd,VIDIOC_DQBUF,&buffer));
                    stop_stream(cd.fd);
                    is_streaming=false;

                    XClearWindow(display,stream_button);
                    XDrawString(display,stream_button,gc,10,20,"start stream");

                }
                if(!is_streaming){
                    if(cd.valid) deinit_cam(&cd);
                    cd=init_cam(devs_info[i].name);

                    if(cd.valid){
                        start_stream(cd.fd);
                        is_streaming=true;

                        XClearWindow(display,stream_button);
                        char *button_text="stop stream";
                        XDrawString(display,stream_button,gc,10,20,"stop stream");

                        draw_cam_info(display,info_window,gc,&cd);
                    }
                }

            }
            if(e.xbutton.window==gl_window){
                if(e.xbutton.button==Button1){
                    printf("mouse down");
                    zoomer_begin_pan(&zoomer,center);
                }else if(e.xbutton.button==Button4){
                    printf("mouse wheel up\n");
                    zoomer_do_zoom(&zoomer,center,1.1);
                }else if(e.xbutton.button==Button5){
                    printf("mouse wheel down\n");
                    zoomer_do_zoom(&zoomer,center,.9);
                }
            }
        }
        if(XCheckTypedEvent(display,ButtonRelease,&e)){
            if(e.xbutton.window==gl_window){
                if(e.xbutton.button==Button1){
                    printf("mouse up");
                    zoomer_end_pan(&zoomer);
                }
            }
        }

        int idx=0;
        if(is_streaming){
            
            struct v4l2_buffer buffer={0};
            buffer.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory=V4L2_MEMORY_MMAP;
            if(-1==ioctl(cd.fd,VIDIOC_DQBUF,&buffer)){
                puts("loop");
                if(errno!=EAGAIN){
                    PRINT(errno,"%d");
                    goto app_done;
                }
            }

            idx=buffer.index;
            assert(idx==0);
            memset(&buffer,0,sizeof buffer);
            buffer.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory=V4L2_MEMORY_MMAP;
            buffer.index=idx;

            SYSCALL(ioctl(cd.fd,VIDIOC_QBUF,&buffer));
            yuv_to_rgb(cd.buf_ptr,cd.draw_buf,cd.w,cd.h,cd.fmt.fmt.pix.pixelformat);
        }

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        glPushMatrix();
        glViewport(0,0,win_w,win_h);
        
        glScalef(aspect,1,1);
        if(cd.valid) glTexSubImage2D(
            GL_TEXTURE_2D,0,0,0,
            cd.w,
            cd.h,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            cd.draw_buf
        );

        zoomer_repeat_pan(&zoomer,center);

        stretch_blt(zoomer.screen_rect.corner,
                    zoomer.screen_rect.size,
                    zoomer.image_rect.corner,
                    zoomer.image_rect.size);

        glPointSize(5);
        glBegin(GL_POINTS);
        glVertex3f(mouse[0],mouse[1],0);
        glEnd();

        glPopMatrix();
        GL_CHECK();
        glXSwapBuffers(display,gl_window);
        XFlush(display);
        


        print_fps();

    }
app_done:
    if(is_streaming) stop_stream(cd.fd);
    SYSCALL(close(cd.fd));

    XDestroyWindow(display,window);
    XCloseDisplay(display);
    puts("done");

}














//