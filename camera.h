#pragma once
#include "common.h"


#define NUM_BUFS 10

typedef struct{
    char dev_name[16];
    int fd;
    int w;
    int h;
    void *buf_ptrs[NUM_BUFS];
    int buf_size;
    void *draw_buf;
    void *yuv_buf;
    uint32_t tex_id;
    bool valid;

    struct v4l2_capability cap;
    struct v4l2_fmtdesc fmtdesc;
    struct v4l2_format fmt;
}cam_data_t;


void deinit_cam(cam_data_t *cd);
void draw_cam_info(Display *display,Window info_window,GC gc,cam_data_t *cd);
cam_data_t init_cam(const char *dev_name,bool use_yuv);

void enqueue_buf(int fd,int idx);
int dequeue_buf(int fd);


void enqueue_all(int fd);
void dequeue_all(int fd);
