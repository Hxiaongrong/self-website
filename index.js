const vertexShader = `
uniform float iTime;
uniform vec2 iResolution;
uniform vec2 iMouse;

varying vec2 vUv;

void main(){
    vec3 p=position;
    gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
    
    vUv=uv;
}
`;

const fragmentShader = `
uniform vec2 uMouse1;
uniform vec2 uMouse2;
uniform float uSize;
uniform vec2 uAspect;
uniform float uRt2Opacity;

uniform samplerCube uCubemap;
uniform sampler2D uRt;
uniform sampler2D uRt2;

#define SHOW_ISOLINE 0

// consts
const float PI=3.14159265359;

const float TWO_PI=6.28318530718;

// utils
float map(float value,float min1,float max1,float min2,float max2){
    return min2+(value-min1)*(max2-min2)/(max1-min1);
}

// sdf ops
float opUnion(float d1,float d2)
{
    return min(d1,d2);
}

vec2 opUnion(vec2 d1,vec2 d2)
{
    return(d1.x<d2.x)?d1:d2;
}

float opSmoothUnion(float d1,float d2,float k)
{
    float h=max(k-abs(d1-d2),0.);
    return min(d1,d2)-h*h*.25/k;
}

// ray
vec2 normalizeScreenCoords(vec2 screenCoord,vec2 resolution,vec2 aspect)
{
    vec2 uv=screenCoord/resolution.xy;
    uv-=vec2(.5);
    uv*=aspect;
    return uv;
}

mat3 setCamera(in vec3 ro,in vec3 ta,float cr)
{
    vec3 cw=normalize(ta-ro);
    vec3 cp=vec3(sin(cr),cos(cr),0.);
    vec3 cu=normalize(cross(cw,cp));
    vec3 cv=(cross(cu,cw));
    return mat3(cu,cv,cw);
}

vec3 getRayDirection(vec2 p,vec3 ro,vec3 ta,float fl){
    mat3 ca=setCamera(ro,ta,0.);
    vec3 rd=ca*normalize(vec3(p,fl));
    return rd;
}

// lighting
// https://learnopengl.com/Lighting/Basic-Lighting

float saturate(float a){
    return clamp(a,0.,1.);
}

float diffuse(vec3 n,vec3 l){
    float diff=saturate(dot(n,l));
    return diff;
}

float specular(vec3 n,vec3 l,float shininess){
    float spec=pow(saturate(dot(n,l)),shininess);
    return spec;
}

float fresnel(float bias,float scale,float power,vec3 I,vec3 N)
{
    return bias+scale*pow(1.+dot(I,N),power);
}

// rotate
mat2 rotation2d(float angle){
    float s=sin(angle);
    float c=cos(angle);
    
    return mat2(
        c,-s,
        s,c
    );
}

mat4 rotation3d(vec3 axis,float angle){
    axis=normalize(axis);
    float s=sin(angle);
    float c=cos(angle);
    float oc=1.-c;
    
    return mat4(
        oc*axis.x*axis.x+c,oc*axis.x*axis.y-axis.z*s,oc*axis.z*axis.x+axis.y*s,0.,
        oc*axis.x*axis.y+axis.z*s,oc*axis.y*axis.y+c,oc*axis.y*axis.z-axis.x*s,0.,
        oc*axis.z*axis.x-axis.y*s,oc*axis.y*axis.z+axis.x*s,oc*axis.z*axis.z+c,0.,
        0.,0.,0.,1.
    );
}

vec2 rotate(vec2 v,float angle){
    return rotation2d(angle)*v;
}

vec3 rotate(vec3 v,vec3 axis,float angle){
    return(rotation3d(axis,angle)*vec4(v,1.)).xyz;
}

mat3 rotation3dX(float angle){
    float s=sin(angle);
    float c=cos(angle);
    
    return mat3(
        1.,0.,0.,
        0.,c,s,
        0.,-s,c
    );
}

vec3 rotateX(vec3 v,float angle){
    return rotation3dX(angle)*v;
}

mat3 rotation3dY(float angle){
    float s=sin(angle);
    float c=cos(angle);
    
    return mat3(
        c,0.,-s,
        0.,1.,0.,
        s,0.,c
    );
}

vec3 rotateY(vec3 v,float angle){
    return rotation3dY(angle)*v;
}

mat3 rotation3dZ(float angle){
    float s=sin(angle);
    float c=cos(angle);
    
    return mat3(
        c,s,0.,
        -s,c,0.,
        0.,0.,1.
    );
}

vec3 rotateZ(vec3 v,float angle){
    return rotation3dZ(angle)*v;
}

// gamma
const float gamma=2.2;

float toGamma(float v){
    return pow(v,1./gamma);
}

vec2 toGamma(vec2 v){
    return pow(v,vec2(1./gamma));
}

vec3 toGamma(vec3 v){
    return pow(v,vec3(1./gamma));
}

vec4 toGamma(vec4 v){
    return vec4(toGamma(v.rgb),v.a);
}

// sdf
float sdSphere(vec3 p,float s)
{
    return length(p)-s;
}

// noise
//
// GLSL textureless classic 3D noise "cnoise",
// with an RSL-style periodic variant "pnoise".
// Author:  Stefan Gustavson (stefan.gustavson@liu.se)
// Version: 2011-10-11
//
// Many thanks to Ian McEwan of Ashima Arts for the
// ideas for permutation and gradient selection.
//
// Copyright (c) 2011 Stefan Gustavson. All rights reserved.
// Distributed under the MIT license. See LICENSE file.
//

vec3 mod289(vec3 x)
{
    return x-floor(x*(1./289.))*289.;
}

vec4 mod289(vec4 x)
{
    return x-floor(x*(1./289.))*289.;
}

vec4 permute(vec4 x)
{
    return mod289(((x*34.)+1.)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159-.85373472095314*r;
}

vec3 fade(vec3 t){
    return t*t*t*(t*(t*6.-15.)+10.);
}

// Classic Perlin noise
float cnoise(vec3 P)
{
    vec3 Pi0=floor(P);// Integer part for indexing
    vec3 Pi1=Pi0+vec3(1.);// Integer part + 1
    Pi0=mod289(Pi0);
    Pi1=mod289(Pi1);
    vec3 Pf0=fract(P);// Fractional part for interpolation
    vec3 Pf1=Pf0-vec3(1.);// Fractional part - 1.0
    vec4 ix=vec4(Pi0.x,Pi1.x,Pi0.x,Pi1.x);
    vec4 iy=vec4(Pi0.yy,Pi1.yy);
    vec4 iz0=Pi0.zzzz;
    vec4 iz1=Pi1.zzzz;
    
    vec4 ixy=permute(permute(ix)+iy);
    vec4 ixy0=permute(ixy+iz0);
    vec4 ixy1=permute(ixy+iz1);
    
    vec4 gx0=ixy0*(1./7.);
    vec4 gy0=fract(floor(gx0)*(1./7.))-.5;
    gx0=fract(gx0);
    vec4 gz0=vec4(.5)-abs(gx0)-abs(gy0);
    vec4 sz0=step(gz0,vec4(0.));
    gx0-=sz0*(step(0.,gx0)-.5);
    gy0-=sz0*(step(0.,gy0)-.5);
    
    vec4 gx1=ixy1*(1./7.);
    vec4 gy1=fract(floor(gx1)*(1./7.))-.5;
    gx1=fract(gx1);
    vec4 gz1=vec4(.5)-abs(gx1)-abs(gy1);
    vec4 sz1=step(gz1,vec4(0.));
    gx1-=sz1*(step(0.,gx1)-.5);
    gy1-=sz1*(step(0.,gy1)-.5);
    
    vec3 g000=vec3(gx0.x,gy0.x,gz0.x);
    vec3 g100=vec3(gx0.y,gy0.y,gz0.y);
    vec3 g010=vec3(gx0.z,gy0.z,gz0.z);
    vec3 g110=vec3(gx0.w,gy0.w,gz0.w);
    vec3 g001=vec3(gx1.x,gy1.x,gz1.x);
    vec3 g101=vec3(gx1.y,gy1.y,gz1.y);
    vec3 g011=vec3(gx1.z,gy1.z,gz1.z);
    vec3 g111=vec3(gx1.w,gy1.w,gz1.w);
    
    vec4 norm0=taylorInvSqrt(vec4(dot(g000,g000),dot(g010,g010),dot(g100,g100),dot(g110,g110)));
    g000*=norm0.x;
    g010*=norm0.y;
    g100*=norm0.z;
    g110*=norm0.w;
    vec4 norm1=taylorInvSqrt(vec4(dot(g001,g001),dot(g011,g011),dot(g101,g101),dot(g111,g111)));
    g001*=norm1.x;
    g011*=norm1.y;
    g101*=norm1.z;
    g111*=norm1.w;
    
    float n000=dot(g000,Pf0);
    float n100=dot(g100,vec3(Pf1.x,Pf0.yz));
    float n010=dot(g010,vec3(Pf0.x,Pf1.y,Pf0.z));
    float n110=dot(g110,vec3(Pf1.xy,Pf0.z));
    float n001=dot(g001,vec3(Pf0.xy,Pf1.z));
    float n101=dot(g101,vec3(Pf1.x,Pf0.y,Pf1.z));
    float n011=dot(g011,vec3(Pf0.x,Pf1.yz));
    float n111=dot(g111,Pf1);
    
    vec3 fade_xyz=fade(Pf0);
    vec4 n_z=mix(vec4(n000,n100,n010,n110),vec4(n001,n101,n011,n111),fade_xyz.z);
    vec2 n_yz=mix(n_z.xy,n_z.zw,fade_xyz.y);
    float n_xyz=mix(n_yz.x,n_yz.y,fade_xyz.x);
    return 2.2*n_xyz;
}

// blend

float blendScreen(float base,float blend){
    return 1.-((1.-base)*(1.-blend));
}

vec3 blendScreen(vec3 base,vec3 blend){
    return vec3(blendScreen(base.r,blend.r),blendScreen(base.g,blend.g),blendScreen(base.b,blend.b));
}

// color
vec3 saturation(vec3 rgb,float adjustment){
    const vec3 W=vec3(.2125,.7154,.0721);
    vec3 intensity=vec3(dot(rgb,W));
    return mix(intensity,rgb,adjustment);
}

vec4 RGBShift(sampler2D t,vec2 rUv,vec2 gUv,vec2 bUv){
    vec4 color1=texture(t,rUv);
    vec4 color2=texture(t,gUv);
    vec4 color3=texture(t,bUv);
    vec4 color=vec4(color1.r,color2.g,color3.b,color2.a);
    return color;
}

// transforms

vec3 distort(vec3 p){
    float t=iTime*.5;
    
    float distortStr=1.6;
    vec3 distortP=p+cnoise(vec3(p*PI*distortStr+t));
    float perlinStr=cnoise(vec3(distortP*PI*distortStr*.1));
    
    vec3 dispP=p;
    dispP+=(p*perlinStr*.1);
    
    return dispP;
}

vec2 map(in vec3 pos)
{
    vec2 res=vec2(1e10,0.);
    
    // pos=rotate(pos,vec3(1.,1.,1.),iTime);
    
    pos=distort(pos);
    
    vec2 m1=uMouse1.xy;
    m1*=uAspect;
    vec2 m2=uMouse2.xy;
    m2*=uAspect;
    
    {
        float r=uSize;
        vec3 d1p=pos;
        d1p-=vec3(m1*2.,0.);
        float d1=sdSphere(d1p,r);
        vec3 d2p=pos;
        d2p-=vec3(m2*2.,0.);
        float d2=sdSphere(d2p,r-.05);
        d2=opSmoothUnion(d2,d1,.2);
        res=opUnion(res,vec2(d2,114514.));
    }
    
    return res;
}

vec2 raycast(in vec3 ro,in vec3 rd,in float tMax){
    vec2 res=vec2(0.,-1.);
    float t=0.;
    for(int i=0;i<4;i++)
    {
        vec3 p=ro+t*rd;
        vec2 h=map(p);
        if(h.x<.001||t>(tMax+GLOW))
        {
            break;
        };
        t+=h.x;
        res=vec2(t,h.y);
    }
    return res;
}

vec3 calcNormal(vec3 pos,float eps){
    const vec3 v1=vec3(1.,-1.,-1.);
    const vec3 v2=vec3(-1.,-1.,1.);
    const vec3 v3=vec3(-1.,1.,-1.);
    const vec3 v4=vec3(1.,1.,1.);
    
    return normalize(v1*map(pos+v1*eps).x+
    v2*map(pos+v2*eps).x+
    v3*map(pos+v3*eps).x+
    v4*map(pos+v4*eps).x);
}

vec3 calcNormal(vec3 pos){
    return calcNormal(pos,.002);
}

vec3 drawIsoline(vec3 col,vec3 pos){
    float d=map(pos).x;
    col*=1.-exp(-6.*abs(d));
    col*=.8+.2*cos(150.*d);
    col=mix(col,vec3(1.),1.-smoothstep(0.,.01,abs(d)));
    return col;
}

vec3 material(in vec3 col,in vec3 pos,in float m,in vec3 nor){
    // col=vec3(1.);
    col=vec3(0.);
    
    if(m==114514.){
        if(SHOW_ISOLINE==1){
            col=drawIsoline(col,vec3(pos.x*1.,pos.y*0.,pos.z*1.));
        }
    }
    
    return col;
}

vec3 lighting(in vec3 col,in vec3 pos,in vec3 rd,in vec3 nor,in float t,in vec2 screenUv){
    vec3 lin=col;
    
    // diffuse
    // vec3 lig=normalize(vec3(1.,2.,3.));
    // float dif=dot(nor,lig)*.5+.5;
    // lin=col*dif;
    
    vec3 m=vec3(uMouse1*iResolution.xy,0.);
    vec3 viewDir=normalize(vec3(0.)-vec3(m.x/(iResolution.x*.25),m.y/(iResolution.y*.25),-2.));
    vec3 I=normalize(nor.xyz-viewDir);
    float distanceMouse=distance(uMouse1,vec2(0.))*.1;
    
    // refract
    vec3 refra=refract(vec3(0.,0.,-2.),nor,1./2.);
    screenUv+=refra.xy*.015;
    
    // fresnel
    float fOffset=-1.4*(1.-distanceMouse*2.);
    float f=fOffset+fresnel(0.,1.,1.,I,nor)*1.44;
    float f2=fOffset+fresnel(1.,1.,1.,rd,nor)*1.44;
    vec3 fCol=vec3(saturate(pow(f-.8,3.)));
    lin=blendScreen(lin,fCol);
    
    // cube
    vec3 cubeTex=texture(uCubemap,vec3(screenUv,0.)).rgb;
    vec3 cubeTexSat=saturation(cubeTex,6.);
    vec3 cubeTexF=blendScreen(mix(vec3(0.),cubeTexSat,fCol),fCol);
    lin=blendScreen(lin,cubeTexF);
    
    // iridescence
    vec3 iri=vec3(0.);
    float iriSrength=10.;
    iri.r=smoothstep(cubeTexF.r*iriSrength,0.,.5);
    iri.g=smoothstep(cubeTexF.g*iriSrength,0.,.5);
    iri.b=smoothstep(cubeTexF.b*iriSrength,0.,.5);
    lin=blendScreen(lin,iri);
    
    vec3 iri2=vec3(0.);
    iri2.r=smoothstep(0.,.25,cubeTexF.r);
    iri2.g=smoothstep(0.,.25,cubeTexF.r);
    iri2.b=smoothstep(0.,.25,cubeTexF.r);
    lin=blendScreen(lin,iri2);
    
    // middle fresnel
    vec3 mf=vec3(0.);
    float fFactor=pow(f+f2,1.24);
    float invertFFactor=-fFactor+3.;
    mf=vec3(invertFFactor);
    mf*=.1;
    lin=blendScreen(lin,mf);
    
    // rt
    float offset=(.05*nor.x*.15+.002)*.8;
    vec2 rUv=vec2(screenUv.x,screenUv.y+offset);
    vec2 gUv=vec2(screenUv.x,screenUv.y);
    vec2 bUv=vec2(screenUv.x,screenUv.y-offset);
    vec3 rtTex=RGBShift(uRt,rUv,gUv,bUv).xyz;
    lin=blendScreen(lin,rtTex);
    
    // rt2
    vec3 rt2Tex=texture(uRt2,screenUv).xyz;
    lin=blendScreen(lin,rt2Tex*uRt2Opacity);
    
    return lin;
}

vec4 render(in vec3 ro,in vec3 rd,in vec2 screenUv){
    vec4 col=vec4(0.);
    
    float tMax=2.15;
    vec2 res=raycast(ro,rd,tMax);
    float t=res.x;
    float m=res.y;
    
    if(t<tMax){
        vec3 pos=ro+t*rd;
        
        vec3 nor=calcNormal(pos);
        
        vec3 result=vec3(0.);
        result=material(result,pos,m,nor);
        result=lighting(result,pos,rd,nor,t,screenUv);
        
        col=vec4(result,1.);
    }
    
    if(t>tMax&&t<(tMax+GLOW)){
        vec3 glowColor=vec3(1.);
        float glowAlpha=map(t,tMax,tMax+GLOW,1.,0.);
        col=vec4(glowColor,glowAlpha);
    }
    
    return col;
}

vec4 getSceneColor(vec2 fragCoord){
    vec2 p=normalizeScreenCoords(fragCoord,iResolution.xy,uAspect);
    
    vec3 ro=vec3(0.,0.,2.);
    vec3 rd=normalize(vec3(p,-1.));
    
    vec2 screenUv=fragCoord.xy/iResolution.xy;
    
    // render
    vec4 col=render(ro,rd,screenUv);
    
    // gamma
    // col=toGamma(col);
    
    return col;
}

void mainImage(out vec4 fragColor,in vec2 fragCoord){
    vec4 tot=vec4(0.);
    
    float AA_size=1.;
    float count=0.;
    for(float aaY=0.;aaY<AA_size;aaY++)
    {
        for(float aaX=0.;aaX<AA_size;aaX++)
        {
            tot+=getSceneColor(fragCoord+vec2(aaX,aaY)/AA_size);
            count+=1.;
        }
    }
    tot/=count;
    
    fragColor=tot;
}
`;

const vertexShader2 = `
uniform float iTime;
uniform vec2 iResolution;
uniform vec2 iMouse;

varying vec2 vUv;

void main(){
    vec3 p=position;
    gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
    
    vUv=uv;
}
`;

const fragmentShader2 = `
varying vec2 vUv;

uniform vec3 uTextColor;

void main(){
    vec2 p=vUv;
    
    vec4 col=vec4(uTextColor,1.);
    
    gl_FragColor=col;
}
`;

const vertexShader3 = `
uniform float iTime;
uniform vec2 iResolution;
uniform vec2 iMouse;

varying vec2 vUv;

void main(){
    vec3 p=position;
    gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);
    
    vUv=uv;
}
`;

const fragmentShader3 = `
uniform float iTime;
uniform vec2 iResolution;
uniform vec2 iMouse;

uniform sampler2D uTexture;

varying vec2 vUv;

uniform float uOpacity;

void main(){
    vec2 p=vUv;
    vec4 tex=texture(uTexture,p);
    vec3 col=tex.rgb;
    float alpha=uOpacity;
    gl_FragColor=vec4(col,alpha);
}
`;

class LiquidCrystal extends kokomi.Component {
  constructor(base, config) {
    super(base);

    const { scroller } = config;

    this.scroller = scroller;

    const resourceList = [
      {
        name: "cubemap",
        type: "cubeTexture",
        path: [
          "https://s2.loli.net/2022/11/02/AdySfoqhV8W5Fgr.png",
          "https://s2.loli.net/2022/11/02/raZmYvN5kC8gVdu.png",
          "https://s2.loli.net/2022/11/02/jhUc8kHMIxBwKSR.png",
          "https://s2.loli.net/2022/11/02/Dk6grUanARNLpOM.png",
          "https://s2.loli.net/2022/11/02/CwBdbtuMoQmKcjq.png",
          "https://s2.loli.net/2022/11/02/SrZMC3bDAd7xJwj.png",
        ],
      },
    ];
    this.resourceList = resourceList;

    const params = {
      size: 0.28,
      glow: 0.005,
      mouse1Lerp: 0.1,
      mouse2Lerp: 0.09,
    };
    this.params = params;

    this.offsetX1 = 0;
    this.offsetY1 = 0;

    this.offsetX2 = 0;
    this.offsetY2 = 0;

    this.isMouseFollow = true;
    this.moveTargetX = 0;
    this.moveTargetY = 0;
  }
  async addExisting() {
    const { base, params, resourceList } = this;

    return new Promise((resolve) => {
      const am = new kokomi.AssetManager(base, resourceList);
      am.emitter.on("ready", () => {
        const sq = new kokomi.ScreenQuad(base, {
          shadertoyMode: true,
          fragmentShader,
          uniforms: {
            uMouse1: {
              value: new THREE.Vector2(0, 0),
            },
            uMouse2: {
              value: new THREE.Vector2(0, 0),
            },
            uSize: {
              value: params.size,
            },
            uAspect: {
              value: new THREE.Vector2(1, 1),
            },
            uCubemap: {
              value: null,
            },
            uRt: {
              value: null,
            },
            uRt2: {
              value: null,
            },
            uRt2Opacity: {
              value: 0,
            },
          },
        });
        sq.material.transparent = true;
        sq.material.defines = {
          GLOW: params.glow,
        };
        sq.addExisting();

        sq.material.uniforms.uCubemap.value = am.items["cubemap"];

        sq.mesh.position.z = 1;

        this.sq = sq;

        resolve(sq);
      });
    });
  }
  moveTo(x, y) {
    const { params, sq } = this;

    const mouse = new THREE.Vector2(
      x / window.innerWidth,
      y / window.innerHeight
    );

    const mouse1Lerp = params.mouse1Lerp;
    const mouse2Lerp = params.mouse2Lerp;

    this.offsetX1 = THREE.MathUtils.lerp(this.offsetX1, mouse.x, mouse1Lerp);
    this.offsetY1 = THREE.MathUtils.lerp(this.offsetY1, mouse.y, mouse1Lerp);

    this.offsetX2 = THREE.MathUtils.lerp(
      this.offsetX2,
      this.offsetX1,
      mouse2Lerp
    );
    this.offsetY2 = THREE.MathUtils.lerp(
      this.offsetY2,
      this.offsetY1,
      mouse2Lerp
    );

    sq.material.uniforms.uMouse1.value = new THREE.Vector2(
      this.offsetX1,
      this.offsetY1
    );
    sq.material.uniforms.uMouse2.value = new THREE.Vector2(
      this.offsetX2,
      this.offsetY2
    );
  }
  update() {
    const { sq } = this;

    if (sq) {
      // mouse
      if (this.isMouseFollow) {
        this.moveTargetX = this.base.iMouse.mouseScreen.x;
        this.moveTargetY = this.base.iMouse.mouseScreen.y;
      }

      // move
      this.moveTo(this.moveTargetX, this.moveTargetY);

      // aspect
      if (window.innerHeight / window.innerWidth > 1) {
        sq.material.uniforms.uAspect.value = new THREE.Vector2(
          window.innerWidth / window.innerHeight,
          1
        );
      } else {
        sq.material.uniforms.uAspect.value = new THREE.Vector2(
          1,
          window.innerHeight / window.innerWidth
        );
      }

      // rt
      if (this.rt) {
        sq.material.uniforms.uRt.value = this.rt.texture;
      }

      // rt2
      if (this.rt2) {
        sq.material.uniforms.uRt2.value = this.rt2.texture;
      }
    }
  }
  setRt(rt) {
    this.rt = rt;
  }
  setRt2(rt2) {
    this.rt2 = rt2;
  }
  fadeInRt2() {
    const { sq } = this;

    if (this.rt2) {
      gsap.to(sq.material.uniforms.uRt2Opacity, {
        value: 1,
      });
    }
  }
  fadeOutRt2() {
    const { sq } = this;

    if (this.rt2) {
      gsap.to(sq.material.uniforms.uRt2Opacity, {
        value: 0,
      });
    }
  }
  followMouse() {
    this.isMouseFollow = true;
  }
  unfollowMouse() {
    this.isMouseFollow = false;
  }
  snapToPoint(el) {
    const rect = el.getBoundingClientRect();
    const { x, y, width, height } = rect;
    const posScreen = this.base.iMouse.getMouseScreen(
      x + width / 2,
      y + height / 2
    );
    const px = posScreen.x;
    const py = posScreen.y;
    gsap.to(this, {
      moveTargetX: px,
    });
    gsap.to(this, {
      moveTargetY: py,
    });
  }
}

class WebGLText extends kokomi.Component {
  constructor(base, config) {
    super(base);

    const { scroller } = config;

    const mg = new kokomi.MojiGroup(base, {
      vertexShader: vertexShader2,
      fragmentShader: fragmentShader2,
      scroller,
      elList: [...document.querySelectorAll(".webgl-text")],
    });
    this.mg = mg;
  }
  addExisting() {
    this.mg.addExisting();

    this.mg.mojis.forEach((moji) => {
      moji.textMesh.mesh.letterSpacing = 0.05;
    });
  }
}

class WebGLGallery extends kokomi.Component {
  constructor(base, config) {
    super(base);

    const { scroller } = config;

    const gallary = new kokomi.Gallery(base, {
      vertexShader: vertexShader3,
      fragmentShader: fragmentShader3,
      materialParams: {
        transparent: true,
      },
      scroller,
      elList: [...document.querySelectorAll(".webgl-img")],
      isScrollPositionSync: false,
      uniforms: {
        uOpacity: 1,
      },
    });
    this.gallary = gallary;

    this.targetX = 0;
    this.targetY = 0;
  }
  async addExisting() {
    await this.gallary.addExisting();
  }
  hideAll() {
    this.gallary.makuGroup.makus.forEach((maku) => {
      maku.mesh.visible = false;
    });
  }
  connectSwiper(swiper) {
    this.swiper = swiper;
  }
  update() {
    if (this.gallary.makuGroup) {
      // swiper
      if (this.swiper) {
        this.gallary.scroller.scroll.target = -this.swiper.translate;
      }

      // mouse
      this.targetX = THREE.MathUtils.lerp(
        this.targetX,
        this.base.interactionManager.mouse.x / 40,
        0.1
      );
      this.targetY = THREE.MathUtils.lerp(
        this.targetY,
        this.base.interactionManager.mouse.y / 40,
        0.1
      );

      this.gallary.makuGroup.makus.forEach((maku) => {
        // mouse follow
        if (maku.el.dataset["webglMouseFollow"] === "1") {
          this.base.update(() => {
            const offsetX = Number(maku.el.dataset["webglMouseOffsetX"]) || 0;
            const posX = this.targetX + offsetX;
            const offsetY = Number(maku.el.dataset["webglMouseOffsetY"]) || 0;
            const posY = this.targetY + offsetY;
            maku.mesh.position.x = (posX * window.innerWidth) / 2;
            maku.mesh.position.y = (posY * window.innerHeight) / 2;
          });
        }
      });
    }
  }
}

class Sketch extends kokomi.Base {
  async create() {
    const screenCamera = new kokomi.ScreenCamera(this);
    screenCamera.addExisting();

    new kokomi.OrbitControls(this);

    // functions
    const start = async () => {
      document.querySelector(".loader-screen").classList.add("hollow");

      await kokomi.sleep(500);

      gsap.to(".gallery,#sketch", {
        opacity: 1,
      });
    };

    // --swiper--
    const swiper = new Swiper(".swiper", {
      direction: "vertical",
      mousewheel: true,
    });
    window.swiper = swiper;

    // await start();
    // return;

    // --webgl--

    // scroller
    const scroller = new kokomi.NormalScroller(this);
    scroller.scroll.ease = 0.025;
    scroller.listenForScroll();

    // liquid crystal
    const lc = new LiquidCrystal(this, {
      scroller,
    });

    await lc.addExisting();

    // load font
    await kokomi.preloadSDFFont();

    // text
    document.querySelectorAll(".webgl-text").forEach((el) => {
      el.classList.add("opacity-0");
      el.classList.add("select-none");
    });
    const wt = new WebGLText(this, {
      scroller,
    });
    wt.addExisting();

    // rt for text
    const rtScene1 = new THREE.Scene();

    const mojiClones = wt.mg.mojis.map((moji) => {
      const mojiClone = moji.textMesh.mesh.clone();
      mojiClone.origin = moji.textMesh.mesh;
      rtScene1.add(mojiClone);
      return mojiClone;
    });

    const rt = new kokomi.RenderTexture(this, {
      rtScene: rtScene1,
      rtCamera: this.camera,
    });

    this.update(() => {
      lc.setRt(rt);

      if (mojiClones) {
        mojiClones.forEach((mojiClone) => {
          mojiClone.position.copy(mojiClone.origin.position);
        });
      }
    });

    // gallery
    document.querySelectorAll(".webgl-img").forEach((el) => {
      el.classList.add("opacity-0");
    });
    const wg = new WebGLGallery(this, {
      scroller,
    });
    await wg.addExisting();
    wg.connectSwiper(swiper);

    // rt for img
    const rtScene2 = new THREE.Scene();

    const makuClones = wg.gallary.makuGroup.makus.map((maku) => {
      const makuClone = maku.mesh.clone();
      makuClone.origin = maku.mesh;
      makuClone.el = maku.el;
      rtScene2.add(makuClone);
      return makuClone;
    });

    wg.hideAll();

    const rt2 = new kokomi.RenderTexture(this, {
      rtScene: rtScene2,
      rtCamera: this.camera,
    });

    this.update(() => {
      lc.setRt2(rt2);

      if (wg.gallary.makuGroup.makus && makuClones) {
        makuClones.forEach((makuClone) => {
          makuClone.position.copy(makuClone.origin.position);
        });
      }
    });

    const showImgOnly = (id) => {
      makuClones.forEach((maku) => {
        gsap.to(maku.material.uniforms.uOpacity, {
          value: 0,
          duration: 0.8,
        });

        if (id === Number(maku.el.dataset["webglImgId"])) {
          gsap.to(maku.material.uniforms.uOpacity, {
            value: 1,
            duration: 0.8,
          });
        }
      });
    };

    // transition
    let activeIndex = swiper.activeIndex;

    swiper.on("slideChange", (e) => {
      activeIndex = swiper.activeIndex;

      if (activeIndex === 0) {
        lc.followMouse();
        lc.fadeOutRt2();
      } else if (activeIndex === 1) {
        lc.unfollowMouse();
        lc.fadeInRt2();
        lc.snapToPoint(document.querySelector(".webgl-snap-point-1"));
        showImgOnly(1);
      } else if (activeIndex === 2) {
        lc.unfollowMouse();
        lc.fadeInRt2();
        lc.snapToPoint(document.querySelector(".webgl-snap-point-2"));
        showImgOnly(2);
      } else if (activeIndex === 3) {
        lc.followMouse();
        lc.fadeOutRt2();
      }
    });

    // load images
    await wg.gallary.checkImagesLoaded();

    await start();
  }
}

const createSketch = () => {
  const sketch = new Sketch();
  sketch.create();
  return sketch;
};

createSketch();

const makeResponsive = () => {
  const clamp = (num, lower, upper) =>
    Math.max(
      Math.min(Number(num), Math.max(lower, upper)),
      Math.min(lower, upper)
    );

  // 基准宽度
  const baseWidth = 1536;

  // 基准字体大小
  const baseSize = 16;

  // 最小字体大小
  const minSize = 10;

  // 设置 rem 函数
  const setRem = () => {
    // 当前页面宽度相对于基准宽度的缩放比例
    const scale = document.documentElement.clientWidth / baseWidth;
    // 设置目标字体大小
    const target = baseSize * scale;
    // 限制字体大小
    const fontSize = clamp(target, minSize, Infinity);
    document.documentElement.style.fontSize = `${fontSize}px`;
  };

  // 执行setRem
  const doSetRem = () => {
    setRem();
    window.addEventListener("resize", setRem);
  };

  // 还原rem
  const resetRem = () => {
    document.documentElement.style.fontSize = `${16}px`;
    window.removeEventListener("resize", setRem);
  };

  doSetRem();
};

makeResponsive();
