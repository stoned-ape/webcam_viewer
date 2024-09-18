#include "camera.h"

void deinit_cam(cam_data_t *cd){
    puts("start\n");
    if(cd->valid){
        for(int i=0;i<NUM_BUFS;i++) SYSCALL(munmap(cd->buf_ptrs[i],cd->buf_size));
        PRINT(cd->fd,"%d");
        SYSCALL(close(cd->fd));
        // free(cd->draw_buf);
        cudaFree(cd->draw_buf);
        cudaFree(cd->yuv_buf);
        cd->draw_buf=NULL;
        cd->yuv_buf=NULL;
    }
    memset(cd,0,sizeof(cam_data_t));
    cd->fd=-1;
    puts("end");
}

void draw_cam_info(Display *display,Window info_window,GC gc,cam_data_t *cd){
    XClearWindow(display,info_window);
    XDrawString(display,info_window,gc,5 ,20,"selected device:");
    XDrawString(display,info_window,gc,15,40 ,cd->dev_name);
    XDrawString(display,info_window,gc,15,60 ,(const char*)cd->cap.card);
    XDrawString(display,info_window,gc,15,80 ,(const char*)cd->cap.driver);
    XDrawString(display,info_window,gc,15,100,(const char*)cd->cap.bus_info);
    XDrawString(display,info_window,gc,15,120,(const char*)cd->fmtdesc.description);
    x_draw_printf(display,info_window,gc,15,140,"%dx%d",cd->w,cd->h);
}

cam_data_t init_cam(const char *dev_name,bool use_yuv){
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
    //SYSCALL(ioctl(fd,VIDIOC_S_CTRL,&control));

    // struct v4l2_control control;
    control.id=0x009a0902;
    control.value=50;
    // control.value=50;
    if(0!=strcmp((char*)cap.driver,"uvcvideo")) control.value*=1000;
    SYSCALL_NOEXIT(ioctl(fd,VIDIOC_S_CTRL,&control));


    struct v4l2_fmtdesc fmtdesc={0};
    fmtdesc.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmtdesc.index=0;
    int pixel_format=0;
    while(0==ioctl(fd,VIDIOC_ENUM_FMT,&fmtdesc)){
        printf("%s: ",(char*)&fmtdesc.pixelformat);

        const int pf=fmtdesc.pixelformat;

        if(use_yuv){
            if(pf=='YUYV' || pf=='YVYU' || pf=='UYVY' || pf=='VYUY'){
                pixel_format=pf;
                cd.fmtdesc=fmtdesc;
            }
        }else{
            if(pf=='21GB'){
                pixel_format=pf;
                cd.fmtdesc=fmtdesc;
            }
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


    struct v4l2_format fmt={0};
    fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;

    // fmt.fmt.pix.width=1920;
    // fmt.fmt.pix.height=1080;
    // fmt.fmt.pix.width=1280;
    // fmt.fmt.pix.height=720;
    fmt.fmt.pix.width=2432;
    fmt.fmt.pix.height=2048;
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
    reqbuf.count=NUM_BUFS;
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

        cd.buf_ptrs[i]=SYSCALL(mmap(NULL,buffer.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd,buffer.m.offset));
        assert(cd.buf_ptrs[i]);
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
    
    // cd.draw_buf=malloc(4*fmt.fmt.pix.width*fmt.fmt.pix.height);
    cudaMallocManaged(&cd.draw_buf,4*fmt.fmt.pix.width*fmt.fmt.pix.height);
    assert(cd.draw_buf);
    cudaMallocManaged(&cd.yuv_buf,2*fmt.fmt.pix.width*fmt.fmt.pix.height);
    assert(cd.yuv_buf);

    glGenTextures(1, &cd.tex_id);
    assert(cd.tex_id>0);
    glBindTexture(GL_TEXTURE_2D, cd.tex_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cd.w, cd.h, 0, GL_RGBA, GL_UNSIGNED_BYTE,cd.draw_buf);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
    return cd;
}
