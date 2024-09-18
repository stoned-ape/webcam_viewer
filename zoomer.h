#pragma once
#include "common.h"

typedef __attribute__((vector_size(8))) float _float2;

static _float2 make__float2(float x,float y){
    _float2 f={x,y};
    return f;
}

typedef struct{
    _float2 corner;
    _float2 size;
}rect;

typedef struct{
    _float2 image_size;
    _float2 screen_size;
    rect image_rect;
    rect screen_rect;
    float new_scale;
    float old_scale;
    float max_scale;
    float min_scale;
    bool panning;
    _float2 old_center;
}zoomer_t;


void zoomer_init(zoomer_t *self,_float2 _image_size,_float2 _screen_size);
void zoomer_do_zoom(zoomer_t *self,const _float2 center,const float delta_scale);
void zoomer_begin_pan(zoomer_t *self,const _float2 center);
void zoomer_repeat_pan(zoomer_t *self,const _float2 center);
void zoomer_end_pan(zoomer_t *self);
void stretch_blt(_float2 dst_corner,_float2 dst_size,_float2 src_corner,_float2 src_size);
