#pragma once
#include "common.h"

__global__ void unpack(const void *packed_data,void *unpacked_data,int n);

__global__ void bayer_to_rgb(
        const void *bayer_data,void *rgb_data,
        int w,int h);

__global__ void yuv_to_rgb(
        const void *yuv_data,void *rgb_data,
        int width,int height,unsigned int fmt);
    