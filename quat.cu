#include "quat.h"

void glRotate(_quat q){glRotatef(deg(get_angle(q)),q.v.x,q.v.y,q.v.z);}
_quat quat_conj(_quat q){return _quat{q.s,-q.v};}
_quat operator*(const _quat l,const _quat r){
    return _quat{l.s*r.s-dot(l.v,r.v),l.s*r.v+r.s*l.v+cross(l.v,r.v)};
}
_quat &operator*=(_quat &l,const _quat r){
    return l=l*r;
}
bool operator==(const _quat l,const _quat r){
    auto div=r*quat_conj(l);
    union{uint32_t i;float f;} u;
    u.f=div.s;
    u.i&=~(1<<31);
    return u.f>.99;
}
float get_angle(_quat q){return 2*atan(glm::length(q.v)/q.s);}
_float3x3 to_mtx3(_quat q){
    auto i=_float3(1,0,0);
    auto j=_float3(0,1,0);
    auto k=_float3(0,0,1);
    i=(q*_quat{0,i}*quat_conj(q)).v;
    j=(q*_quat{0,j}*quat_conj(q)).v;
    k=(q*_quat{0,k}*quat_conj(q)).v;
    _float3x3 _mtx=_float3x3(
        i.x,i.y,i.z,
        j.x,j.y,j.z,
        k.x,k.y,k.z
    );
    return _mtx;
}
_float4x4 to_mtx(_quat q){
    return _float4x4(to_mtx3(q));
    // auto i=_float3(1,0,0);
    // auto j=_float3(0,1,0);
    // auto k=_float3(0,0,1);
    // i=(q*_quat{0,i}*quat_conj(q)).v;
    // j=(q*_quat{0,j}*quat_conj(q)).v;
    // k=(q*_quat{0,k}*quat_conj(q)).v;
    // _float4x4 _mtx=_float4x4(
    //     i.x,i.y,i.z,0.,
    //     j.x,j.y,j.z,0.,
    //     k.x,k.y,k.z,0.,
    //     0. ,0. ,0. ,1.
    // );
    // return _mtx;
}

_float3 rotate(_float3 v,_quat q){
    return (q*_quat{0,v}*quat_conj(q)).v;
}
_quat angle_axis(float theta,_float3 n){
    return _quat{cos(theta/2),sin(theta/2)*glm::normalize(n)};
}
_quat from_to(_float3 a,_float3 b){
    _float3 axis=glm::cross(a,b);
    float angle=acos(glm::dot(glm::normalize(a),glm::normalize(b)));
    float mag=glm::length(axis);
    if(mag<1e-6) return _quat{1,_float3(0)};
    return angle_axis(angle,axis/mag);
}