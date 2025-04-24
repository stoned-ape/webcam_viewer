// compile on linux: gcc x11_opengl.c -lm -lGL -lX11
// compile on mac:   clang -framework OpenGL -I /opt/X11/include/ -L /opt/X11/lib/ -lGL -lX11 x11_opengl.c 
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#if defined(__APPLE__)
_Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"");
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <GL/glx.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <sys/timeb.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "quat.h"

#define PRINT(val,type) printf("%s:\t"type"\n",#val,val);


// const float pi=3.141592653589793;

// double itime(){
//     struct timeb now;
//     ftime(&now);
//     return (double)(now.time%(60*60*24))+now.millitm/1e3;
// }

// void print_fps(){
//     static double timer;
//     double delta=itime()-timer;
//     timer+=delta;
//     printf("\rfps = %f ",1/delta);
//     fflush(stdout);
// }

// float map(float t,float t0,float t1,float s0,float s1){
//     return s0+(s1-s0)*(t-t0)/(t1-t0);
// }

void color_wheel(float t,float ret[4]){
    float theta=map(t,0.,1.,0.,2.*pi);
    float modulus=2.0f*pi*.75f;
    theta=fmodf(theta,modulus)+2.*pi*.87;
    float angles[2]={cosf(theta),sin(theta)};
    angles[0]=map(angles[0],-1.,1.,0.,pi/2.);
    angles[1]=map(angles[1],-1.,1.,0.,pi/2.);
    ret[0]=cosf(angles[0])*sin(angles[1]);
    ret[1]=cosf(angles[1]);
    ret[2]=sin(angles[0])*sin(angles[1]);
    ret[3]=1;
}

void draw_square(){
    glBegin(GL_TRIANGLE_STRIP);
    glVertex3f(-1,-1,0);
    glVertex3f(-1,+1,0);
    glVertex3f(+1,-1,0);
    glVertex3f(+1,+1,0);
    glEnd();
}

// float pos(bool p){return p?1.0f:-1.0f;}

void draw_test_cube(){
    glScalef(.75,.75,.75);
    for(int k=0;k<3;k++){
        float rgb[3]={k==0,k==1,k==2};
        for(int j=0;j<2;j++){
            if(j==0) glColor3fv(rgb);
            else glColor3f(1-rgb[0],1-rgb[1],1-rgb[2]);
            glBegin(GL_TRIANGLE_STRIP);
            float vtx[3];
            vtx[k]=.5*pos(j);
            for(int i=0;i<4;i++){
                vtx[(k+1)%3]=+.5*pos(i/2);
                vtx[(k+2)%3]=+.5*pos(i%2);
                glVertex3fv(vtx);
            }
            glEnd();
        }
    }
}

void draw_test_octa(){
    for(int i=0;i<8;i++){
        glColor3f(i&1,i&2,i&4);
        glBegin(GL_TRIANGLES);
        glVertex3f(.5*pos(i&1),0,0);
        glVertex3f(0,.5*pos(i&2),0);
        glVertex3f(0,0,.5*pos(i&4));
        glEnd();
    }
}

void draw_test_tetra(){
    const float a=sqrtf(2)/4;
    float points[4][3]={
        {+a,+.25,+0},
        {-a,+.25,+0},
        {+0,-.25,+a},
        {+0,-.25,-a},
    };
    for(int i=0;i<4;i++){
        glColor3f(points[i][0]+.5,points[i][2]+.5,points[i][2]+.5);
        glBegin(GL_TRIANGLES);
        for(int j=0;j<4;j++) if(i!=j){
            glVertex3fv(points[j]);
        }
        glEnd();
    }
}

void draw_test_ico() {
    float a=2*sinf(pi/5);
    float b=a*sinf(acosf(1/a));
    float d=2*sinf(pi/10);
    float c=sqrtf(a*a-d*d)/2;
    float s=.5;
    float ud[2][4]={{0,s*(b+c),0,s},{0,s*(-b-c),0,s}};
    float mid[2][5][4];
    for(int i=0;i<2;i++){
        for(int j=0;j<5;j++){
            float theta=2*pi*j/5.0+pi/5.0*i;
            float p[4]={s*cosf(theta),-s*pos(i)*c,s*sinf(theta),s};
            memcpy(mid[i][j],p,sizeof(p));
        }
    }
    glBegin(GL_TRIANGLES);
    for(int i=0;i<2;i++){
        for(int j=0;j<5;j++){
            float col[4];
            color_wheel(j/6.0+i*1/10.0,col);
            glColor3f(col[0],col[1],col[2]);
            glVertex3fv(mid[i][j]);
            glVertex3fv(mid[i][(j+1)%5]);
            glVertex3fv(ud[i]);
        }
    }
    for(int j=0;j<5;j++){
        for(int i=0;i<2;i++){
            float col[4];
            color_wheel(j/6.0+1/20.0+1/40.0*i,col);
            glColor3f(col[0],col[1],col[2]);
            glVertex3fv(mid[i][j]);
            glVertex3fv(mid[i][(j+(i?4:1))%5]);
            glVertex3fv(mid[!i][j]);
        }
    }
    glEnd();
}

// float deg(float rad){return rad*180/pi;}

void draw_pentagon(){
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0,0,0);
    for(int i=0;i<6;i++){
        float theta=i*2*pi/5;
        glVertex3f(sinf(theta),0,cosf(theta));
    }
    glEnd();
}

void draw_test_dodec(){
    const float s=.4;
    const float v=s/(2*sinf(pi/5));
    const float m=v*cosf(pi/5);
    const float a=2*v*sinf(2*pi/5);
    const float phi=(1+sqrtf(5))/2;
    const float gamma=2*atanf(phi);
    const float h=s*sqrtf((25+11*sqrtf(5))/10);

    int face_idx=0;
    float col[4];

    for(int j=0;j<2;j++){
        glPushMatrix();
        glRotatef(deg(j*pi/5),0,1,0);
        glScalef(1,pos(j),1);
        glTranslatef(0,h/2,0);
        glPushMatrix();
        glScalef(v,v,v);
        color_wheel(((7*face_idx++)%12)/12.0+1/48.0,col);
        glColor3fv(col);
        draw_pentagon();
        glPopMatrix();
        for(int i=0;i<5;i++){
            glPushMatrix();
            glRotatef(deg(2*i*pi/5),0,1,0);
            glTranslatef(0,0,-m);
            glRotatef(deg(gamma),1,0,0);
            glTranslatef(0,0,m);
            glScalef(v,v,v);
            color_wheel(((7*face_idx++)%12)/12.0+1/48.0,col);
            glColor3fv(col);
            draw_pentagon();
            glPopMatrix();
        }
        glPopMatrix();
    }
}



void draw_test_sphere(int res){
    float inc=pi/res;
    float rho=.5;
    for(float phi=0;phi<pi;phi+=inc){
        for(float theta=0;theta<2*pi;theta+=inc){
            glBegin(GL_TRIANGLE_STRIP);
            glColor3f(
                .5+.5*cosf(theta+inc/2)*sinf(phi+inc/2),
                .5+.5*                  cosf(phi+inc/2),
                .5+.5*sinf(theta+inc/2)*sinf(phi+inc/2)
            );
            for(int i=0;i<4;i++){
                float t=theta+inc*(i&1);
                float p=phi+.5*inc*(i&2);
                glVertex3f(
                    rho*cosf(t)*sinf(p),
                    rho*cosf(p),
                    rho*sinf(t)*sinf(p)
                );
            }
            glEnd();
        }
    }
}

int main(){
    puts(__func__);
    
    Display *display=XOpenDisplay(NULL);
    assert(display);

    Window root=DefaultRootWindow(display);

    int att[]={GLX_RGBA,GLX_DEPTH_SIZE,24,GLX_DOUBLEBUFFER,None};
    XVisualInfo *vi=glXChooseVisual(display, 0, att);
    assert(vi);

    XSetWindowAttributes swa;
    Colormap cmap=XCreateColormap(display,root,vi->visual,AllocNone);
    swa.colormap=cmap;
    swa.event_mask=ExposureMask|KeyPressMask;
    int width=600,height=600;
    Window win=XCreateWindow(display,root,0,0,width,height,0,vi->depth,
                        InputOutput,vi->visual,CWColormap|CWEventMask,&swa);
    XMapWindow(display,win);
    XStoreName(display,win, ".______.");

    GLXContext glc=glXCreateContext(display,vi,NULL,GL_TRUE);
    glXMakeCurrent(display,win,glc);

    puts((const char *)glGetString(GL_VERSION));
    glEnable(GL_DEPTH_TEST);

    _quat q{1.0f,_float3(0)};

    
    libusb_device_handle *stm_connect();
    _float2x3 read_gyroscope(libusb_device_handle *dh);

    libusb_device_handle *dh=stm_connect();
    float t=itime();
    // _float3 angvel0(0);
    // for(int i=0;i<100;i++) angvel0+=read_gyroscope(dh);
    // angvel0/=100;

    bool running=true;
    while(running){     
        XEvent e={0};

        while(XCheckWindowEvent(display,win,ExposureMask|KeyPressMask,&e)){
            puts("event");
        }

        int root_x,root_y,win_x,win_y;
        Window child;
        unsigned int mask;
        XQueryPointer(display,root,&root,&child,&root_x,&root_y,&win_x,&win_y,&mask);
        float theta=map(root_x,0,width,pi,-pi);
        float phi=map(root_y,0,height,pi/2,-pi/2);

        float t1=itime();
        float dt=t1-t;
        t=t1;


        _float2x3 imu_vals=read_gyroscope(dh);
        _float3 angvel=imu_vals[0];
        _float3 acc=imu_vals[1];

        // angvel.x*=-1;
        // angvel.z*=-1;
        // angvel.x=0;

        // angvel.x=0;
        // angvel.y=0;
        // angvel.z=0;

        // angvel*=.1;

        println(angvel);

        _quat qx=_quat{cosf(.5*angvel.x*dt),_float3(1,0,0)*sinf(.5*angvel.x*dt)};
        _quat qy=_quat{cosf(.5*angvel.y*dt),_float3(0,1,0)*sinf(.5*angvel.y*dt)};
        _quat qz=_quat{cosf(.5*angvel.z*dt),_float3(0,0,1)*sinf(.5*angvel.z*dt)};

        q=qx*qy*qz*q;


        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        for(int i=0;i<4;i++){
            glViewport((i&1)*width/2,(i>1)*height/2,width/2,height/2);
            glPushMatrix();
            glScalef(height/width,1,1);
            // glRotatef(deg(phi),1,0,0);
            // glRotatef(deg(theta),0,1,0);


            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex<3>(acc);
            glEnd();
            
            glRotate(q);

            switch(i){
                case 0:draw_test_octa();break;
                case 1:draw_test_cube();break;
                case 2:draw_test_ico();break;
                case 3:draw_test_dodec();break;
            }
            // draw_test_sphere(20);
            glPopMatrix();
        }
        assert(glGetError()==GL_NO_ERROR);
        glXSwapBuffers(display, win);
        // print_fps();
    } 
    XDestroyWindow(display,win);
    XCloseDisplay(display);   
}