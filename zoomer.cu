#include "zoomer.h"


_float2 map2(_float2 t,_float2 t0,_float2 t1,_float2 s0,_float2 s1){
    return s0+(s1-s0)*(t-t0)/(t1-t0);
}


void zoomer_init(zoomer_t *self,_float2 _image_size,_float2 _screen_size){
    self->panning=false;
    self->new_scale=1;
    self->old_scale=1;
    self->image_size=_image_size;
    self->screen_size=_screen_size;
    self->image_rect.corner=make__float2(0,0);
    self->screen_rect.corner=make__float2(0,0);
    self->image_rect.size=_image_size;
    self->screen_rect.size=_screen_size;
    self->max_scale=200;
    self->min_scale=1;
    self->old_center=make__float2(.5f,.5f);
}

void zoomer_do_zoom(zoomer_t *self,const _float2 center,const float delta_scale){
    self->new_scale=clamp(self->new_scale*delta_scale,self->min_scale,self->max_scale);

    const _float2 screen_tl=self->screen_rect.corner;
    const _float2 screen_br=self->screen_rect.corner+self->screen_rect.size;

    _float2 image_tl=self->image_rect.corner;
    _float2 image_br=self->image_rect.corner+self->image_rect.size;

    const _float2 image_center=map2(center,screen_tl,screen_br,image_tl,image_br);

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

void zoomer_begin_pan(zoomer_t *self,const _float2 center){
    self->panning=true;
    self->old_center=center;
}

void zoomer_repeat_pan(zoomer_t *self,const _float2 center){
    if(!self->panning) return;
    self->image_rect.corner-=(center-self->old_center)/self->new_scale;
    zoomer_do_zoom(self,center,1);
}

void zoomer_end_pan(zoomer_t *self){
    self->panning=false;
}

void glTexCoord(_float2 v){
    glTexCoord2fv((float*)&v);
}
void glVertex(_float2 v){
    glVertex2fv((float*)&v);
}

void stretch_blt(_float2 dst_corner,_float2 dst_size,_float2 src_corner,_float2 src_size){
    glPushMatrix();
    glScalef(1,-1,1);
    glTranslatef(-1,-1,0);
    glScalef(2,2,1);

    glEnable(GL_TEXTURE_2D);
    glColor3f(1,1,1);
    glBegin(GL_TRIANGLE_STRIP);

    glTexCoord(src_corner+src_size*make__float2(0,0));
    glVertex(  dst_corner+dst_size*make__float2(0,0));

    glTexCoord(src_corner+src_size*make__float2(0,1));
    glVertex(  dst_corner+dst_size*make__float2(0,1));

    glTexCoord(src_corner+src_size*make__float2(1,0));
    glVertex(  dst_corner+dst_size*make__float2(1,0));

    glTexCoord(src_corner+src_size*make__float2(1,1));
    glVertex(  dst_corner+dst_size*make__float2(1,1));

    glEnd();
    glDisable(GL_TEXTURE_2D);

    glPopMatrix();
}
