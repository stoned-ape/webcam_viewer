#include "common.h"
#include "kernels.h"
#include "zoomer.h"
#include "camera.h"
#include "cv_debayer.h"

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


int start_stream(int fd){
    
    enum v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    return SYSCALL(ioctl(fd,VIDIOC_STREAMON,&type));
}

int set_trigger_mode(int fd,int mode){
    struct v4l2_control control;
    control.id=0x009a092d;
    control.value=mode;// 0,1,2
    return SYSCALL(ioctl(fd,VIDIOC_S_CTRL,&control));
}

int stop_stream(int fd){
    // set_trigger_mode(fd,0);
    enum v4l2_buf_type type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    return SYSCALL(ioctl(fd,VIDIOC_STREAMOFF,&type));
}





int main(){

    bool trigger_mode=false;
    //108
    //32
    const unsigned int gpio_pin=108;
    struct gpiod_chip *chip=NULL;
    struct gpiod_line *line=NULL;


    chip=gpiod_chip_open_by_name("gpiochip0");
    assert(chip);
    line=gpiod_chip_get_line(chip, gpio_pin);
    assert(line);
    gpiod_line_set_value(line,0);
    gpiod_line_request_output(line, "example", 0);

   

    static struct{char name[16];char card[32];} devs_info[10]={0};
    int devs_count=0;
    int devs_select=1;
    bool use_yuv=true;

    #define MAX_DEVS 10
    // (sizeof(devs_info)/sizeof(devs_info[0]))

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


    int win_w=2432/2;
    int win_h=2048/2;
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

    int button_h=40;
    int button_space=10;
    int button_off=10;
    
    
    Window stream_button=XCreateSimpleWindow(
        display,
        window,
        10,button_off,
        side_bar_w-20,button_h,
        1,
        0x0,0x00aaaaaa
    );
    XMapWindow(display,stream_button);
    XSelectInput(display,stream_button,ExposureMask|ButtonPressMask);

    Window save_button=XCreateSimpleWindow(
        display,
        window,
        10,button_off+1*(button_h+button_space),
        side_bar_w-20,button_h,
        1,
        0x0,0x00aaaaaa
    );
    XMapWindow(display,save_button);
    XSelectInput(display,save_button,ExposureMask|ButtonPressMask);

    Window bayer_button=XCreateSimpleWindow(
        display,
        window,
        10,button_off+2*(button_h+button_space),
        side_bar_w-20,button_h,
        1,
        0x0,0x00aaaaaa
    );
    XMapWindow(display,bayer_button);
    XSelectInput(display,bayer_button,ExposureMask|ButtonPressMask);


    Window dev_buttons[MAX_DEVS];
    for(int i=0;i<devs_count;i++){
        dev_buttons[i]=XCreateSimpleWindow(
            display,
            window,
            10,10+button_off+(3+i)*(button_h+button_space),
            side_bar_w-20,button_h,
            1,
            0x0,0x00aaaaaa
        );
        XMapWindow(display,dev_buttons[i]);
        XSelectInput(display,dev_buttons[i],ExposureMask|ButtonPressMask);
        
    }

    Window info_window=XCreateSimpleWindow(
        display,
        window,
        10,10+button_off+(3+devs_count)*(button_h+button_space),
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

    cam_data_t cd=init_cam(devs_info[devs_select].name,use_yuv);

    zoomer_t zoomer;
    zoomer_init(&zoomer,make__float2(1,1),make__float2(1,1));


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
        _float2 mouse;
        mouse[0]=map(win_x,0,win_w,-1/aspect,1/aspect);
        mouse[1]=map(win_y,0,win_h,1,-1);
        
        const float margin=(1-1/aspect)/2;
        _float2 center;
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
            XDrawString(display,bayer_button,gc,10,20,"use BayerBG12");
            XDrawString(display,window,gc,10,5+button_off+3*(button_h+button_space),"choose video device:");

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
                    stop_stream(cd.fd);
                    dequeue_all(cd.fd);
                    is_streaming=false;

                    XClearWindow(display,stream_button);
                    XDrawString(display,stream_button,gc,10,20,"start stream");

                }else if(cd.valid){ 
                    enqueue_all(cd.fd);
                    start_stream(cd.fd);
                    is_streaming=true;

                    XClearWindow(display,stream_button);
                    char *button_text="stop stream";
                    XDrawString(display,stream_button,gc,10,20,"stop stream");

                }
            }else if(e.xbutton.window==save_button && cd.valid){
                printf("saving image\n");

                if(use_yuv){
                    make_bmp("img.bmp",cd.w,cd.h,cd.draw_buf);
                }else{
                    cv_save("img.tiff",cd.w,cd.h,cd.draw_buf);
                }

            }else if(e.xbutton.window==bayer_button && cd.valid){
                XClearWindow(display,bayer_button);
                if(use_yuv){
                    XDrawString(display,bayer_button,gc,10,20,"use YUV");
                    use_yuv=false;
                }else{
                    XDrawString(display,bayer_button,gc,10,20,"use BayerBG12");
                    use_yuv=true;
                }
                if(is_streaming){ 
                    stop_stream(cd.fd);
                    dequeue_all(cd.fd);
                    is_streaming=false;

                    XClearWindow(display,stream_button);
                    XDrawString(display,stream_button,gc,10,20,"start stream");

                }
                if(!is_streaming){
                    if(cd.valid) deinit_cam(&cd);
                    cd=init_cam(devs_info[devs_select].name,use_yuv);

                    if(cd.valid){
                        start_stream(cd.fd);
                        is_streaming=true;
                        draw_cam_info(display,info_window,gc,&cd);
                    }
                }
            }

            for(int i=0;i<devs_count;i++) if(e.xbutton.window==dev_buttons[i]){
                printf("device %d button clicked\n",i);
                if(is_streaming){ 
                    stop_stream(cd.fd);
                    dequeue_all(cd.fd);
                    is_streaming=false;

                    XClearWindow(display,stream_button);
                    XDrawString(display,stream_button,gc,10,20,"start stream");

                }
                if(!is_streaming){
                    if(cd.valid) deinit_cam(&cd);
                    cd=init_cam(devs_info[i].name,use_yuv);

                    if(cd.valid){
                        start_stream(cd.fd);
                        is_streaming=true;

                        XClearWindow(display,stream_button);
                        char *button_text="stop stream";
                        XDrawString(display,stream_button,gc,10,20,"stop stream");

                        draw_cam_info(display,info_window,gc,&cd);
                        devs_select=i;
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

        // int idx=0;
        if(is_streaming){

            
            if(trigger_mode){
                usleep(1e4); 
                gpiod_line_set_value(line,1);
                usleep(1e3); 
                gpiod_line_set_value(line,0);
            }

            int idx=dequeue_buf(cd.fd);
            enqueue_buf(cd.fd,idx);


            double t0=itime();

            memcpy(cd.yuv_buf,cd.buf_ptrs[idx],cd.w*cd.h*2);
            int block_size=256;
            int num_blocks=(cd.w*cd.h+block_size-1)/block_size;
            if(use_yuv){
                yuv_to_rgb<<<num_blocks,block_size>>>(
                    cd.yuv_buf,cd.draw_buf,cd.w,cd.h,cd.fmt.fmt.pix.pixelformat);
            }else{
                cv_bayer_to_rgb(cd.yuv_buf,cd.draw_buf,cd.w,cd.h);
                // bayer_to_rgb<<<num_blocks,block_size>>>(
                    // cd.yuv_buf,cd.draw_buf,cd.w,cd.h);
            }
            CUDA(cudaDeviceSynchronize());
            double t1=itime();
            // PRINT(t1-t0,"%f");
            // PRINT(1/(t1-t0),"%f");
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
            GL_UNSIGNED_SHORT,
            // GL_UNSIGNED_BYTE,
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
    //enable trigger mode
    if(trigger_mode){
        set_trigger_mode(cd.fd,0);
    }

    if(is_streaming) stop_stream(cd.fd);
    SYSCALL(close(cd.fd));

    XDestroyWindow(display,window);
    XCloseDisplay(display);

    gpiod_line_release(line);
    gpiod_chip_close(chip);

    puts("done");

}














//
