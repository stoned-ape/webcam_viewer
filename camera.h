#pragma once
#include "common.h"


#define NUM_BUFS 10

typedef struct{
    char dev_name[16];
    int fd;
    int w;
    int h;
    int id;
    void *buf_ptrs[NUM_BUFS];
    int buf_size;
    void *draw_buf;
    void *raw_buf;
    void *unpack_buf;
    int raw_buf_size;
    uint32_t tex_id;
    bool valid;
    bool ref_cam;

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

void set_exposure_ms(int fd,int ms);


#define V4L2_CTRL_CLASS_CAMERA 0x009a0000
#define V4L2_CID_CAMERA_CLASS_BASE (V4L2_CTRL_CLASS_CAMERA | 0x900)
/* Custom Control IDs */
enum ecam_custom_ctrl {
    V4L2_CID_CUSTOM_RED_GAIN = (V4L2_CID_CAMERA_CLASS_BASE+40),
    V4L2_CID_CUSTOM_BLUE_GAIN,
    V4L2_CID_CUSTOM_BLACK_LEVEL,
    V4L2_CID_CUSTOM_TEMP_VALUE,
    V4L2_CID_CUSTOM_STANDBY_MODE,
    V4L2_CID_CUSTOM_CAMERA_MODE,
    V4L2_CID_CUSTOM_STROBE,
    V4L2_CID_CUSTOM_MANUAL_WHITE_BALANCE,
    V4L2_CID_CUSTOM_COLOR_TEMPERATURE,
    V4L2_CID_CUSTOM_CCM_Rr,
    V4L2_CID_CUSTOM_CCM_Rg,
    V4L2_CID_CUSTOM_CCM_Rb,
    V4L2_CID_CUSTOM_CCM_Gr,
    V4L2_CID_CUSTOM_CCM_Gg,
    V4L2_CID_CUSTOM_CCM_Gb,
    V4L2_CID_CUSTOM_CCM_Br,
    V4L2_CID_CUSTOM_CCM_Bg,
    V4L2_CID_CUSTOM_CCM_Bb,
    V4L2_CID_UNIQUE_ID
};
