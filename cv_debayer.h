#pragma once
#include "common.h"

void cv_save(const char *name,int w,int h,void *rgb_data);

void cv_bayer_to_rgb(
        const void *bayer_data,void *rgb_data,
        int w,int h);