#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "RocketX_shaders.h"
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/shader.h"

enum {
    MAX_VERTICES = 12000,
    VERTEX_FLOATS = 6,
    VERTEX_BYTES = VERTEX_FLOATS * 4,
    CLEAR_ARGB = 0xFF101827,
    BUFFERS = 2,
    SWAP_INTERVAL = 1,
    BEGIN_TIMEOUT_MS = 5000
};

static const int32_t DEAD_ZONE = 7000;
static const float PI = 3.14159265358979323846f;

typedef struct { float x, y, z; } Vec3;
typedef struct { float m[16]; } Mat4;
typedef struct { float p[3]; float c[3]; } Vertex;

static volatile uint32_t *gpu_vertices;
static Vertex vertices[MAX_VERTICES];
static uint32_t vertex_count;
static XsdkShader vertex_shader, pixel_shader;
static XsdkDepthStencilState depth_state;
static float car_x, car_z, car_yaw, car_speed, car_y;
static float ball_x = 0.0f, ball_y = 1.15f, ball_z = 5.0f;
static float ball_vx, ball_vy, ball_vz;
static float camera_yaw, camera_pitch = 0.22f;

static Vec3 v3(float x, float y, float z) { return (Vec3){x,y,z}; }
static Vec3 add(Vec3 a, Vec3 b) { return v3(a.x+b.x,a.y+b.y,a.z+b.z); }
static Vec3 sub(Vec3 a, Vec3 b) { return v3(a.x-b.x,a.y-b.y,a.z-b.z); }
static Vec3 mul(Vec3 a, float s) { return v3(a.x*s,a.y*s,a.z*s); }
static float dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static Vec3 cross(Vec3 a, Vec3 b) { return v3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
static Vec3 norm(Vec3 a) { float n=sqrtf(dot(a,a)); return n>0.0001f?mul(a,1.0f/n):v3(0,0,0); }
static float clampf(float x,float lo,float hi) { return x<lo?lo:(x>hi?hi:x); }
static float stick(int16_t x) { return (x>DEAD_ZONE||x< -DEAD_ZONE)?(float)x/32768.0f:0.0f; }

static Mat4 identity(void)
{
    Mat4 r={{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}};
    return r;
}

static Mat4 multiply(const Mat4 *a,const Mat4 *b)
{
    Mat4 r;
    for(int c=0;c<4;c++) for(int row=0;row<4;row++) {
        float s=0;
        for(int k=0;k<4;k++) s+=a->m[k*4+row]*b->m[c*4+k];
        r.m[c*4+row]=s;
    }
    return r;
}

static Mat4 camera_matrix(float aspect)
{
    Vec3 target=v3(car_x,0.65f+car_y,car_z);
    float cp=cosf(camera_pitch), sp=sinf(camera_pitch);
    Vec3 offset=v3(sinf(camera_yaw)*cp*8.0f,sp*8.0f,-cosf(camera_yaw)*cp*8.0f);
    Vec3 eye=add(target,offset);
    Vec3 f=norm(sub(target,eye));
    Vec3 s=norm(cross(f,v3(0,1,0)));
    Vec3 u=cross(s,f);
    Mat4 view=identity();
    view.m[0]=s.x; view.m[4]=s.y; view.m[8]=s.z; view.m[12]=-dot(s,eye);
    view.m[1]=u.x; view.m[5]=u.y; view.m[9]=u.z; view.m[13]=-dot(u,eye);
    view.m[2]=-f.x; view.m[6]=-f.y; view.m[10]=-f.z; view.m[14]=dot(f,eye);

    const float near_z=0.1f, far_z=120.0f, fov=1.05f;
    float fs=1.0f/tanf(fov*0.5f);
    float fz=far_z/(near_z-far_z);
    Mat4 projection={{fs/aspect,0,0,0, 0,fs,0,0,
                      0,0,fz,-1, 0,0,near_z*fz,0}};
    return multiply(&projection,&view);
}

static void emit(Vec3 p,Vec3 c)
{
    if(vertex_count>=MAX_VERTICES) return;
    Vertex *v=&vertices[vertex_count++];
    v->p[0]=p.x; v->p[1]=p.y; v->p[2]=p.z;
    v->c[0]=c.x; v->c[1]=c.y; v->c[2]=c.z;
}

static void triangle(Vec3 a,Vec3 b,Vec3 c,Vec3 color)
{
    emit(a,color); emit(b,color); emit(c,color);
}

static void quad(Vec3 a,Vec3 b,Vec3 c,Vec3 d,Vec3 color)
{
    triangle(a,b,c,color); triangle(c,b,d,color);
}

static void box(Vec3 center,Vec3 half,Vec3 color,float yaw)
{
    float cs=cosf(yaw), sn=sinf(yaw);
    Vec3 p[8];
    for(int i=0;i<8;i++) {
        float x=(i&1)?half.x:-half.x;
        float y=(i&2)?half.y:-half.y;
        float z=(i&4)?half.z:-half.z;
        p[i]=add(center,v3(x*cs+z*sn,y,-x*sn+z*cs));
    }
    quad(p[0],p[1],p[2],p[3],color);
    quad(p[5],p[4],p[7],p[6],color);
    quad(p[4],p[0],p[6],p[2],color);
    quad(p[1],p[5],p[3],p[7],color);
    quad(p[2],p[3],p[6],p[7],color);
    quad(p[4],p[5],p[0],p[1],color);
}

static void sphere(Vec3 center,float radius,Vec3 color)
{
    const int rings=10, segments=18;
    for(int y=0;y<rings;y++) {
        float a0=-PI*0.5f+PI*(float)y/rings;
        float a1=-PI*0.5f+PI*(float)(y+1)/rings;
        for(int x=0;x<segments;x++) {
            float b0=2.0f*PI*(float)x/segments;
            float b1=2.0f*PI*(float)(x+1)/segments;
            Vec3 p00=add(center,v3(radius*cosf(a0)*cosf(b0),radius*sinf(a0),radius*cosf(a0)*sinf(b0)));
            Vec3 p01=add(center,v3(radius*cosf(a0)*cosf(b1),radius*sinf(a0),radius*cosf(a0)*sinf(b1)));
            Vec3 p10=add(center,v3(radius*cosf(a1)*cosf(b0),radius*sinf(a1),radius*cosf(a1)*sinf(b0)));
            Vec3 p11=add(center,v3(radius*cosf(a1)*cosf(b1),radius*sinf(a1),radius*cosf(a1)*sinf(b1)));
            quad(p00,p01,p10,p11,color);
        }
    }
}

static void build_arena(void)
{
    const Vec3 turf=v3(0.08f,0.34f,0.25f), wall=v3(0.10f,0.18f,0.30f);
    const Vec3 line=v3(0.85f,0.91f,0.92f);
    box(v3(0,-0.28f,0),v3(18,0.25f,28),turf,0);
    box(v3(0,0.55f,-28),v3(18,0.8f,0.35f),wall,0);
    box(v3(0,0.55f,28),v3(18,0.8f,0.35f),wall,0);
    box(v3(-18,0.55f,0),v3(0.35f,0.8f,28),wall,0);
    box(v3(18,0.55f,0),v3(0.35f,0.8f,28),wall,0);
    box(v3(0,-0.01f,0),v3(0.10f,0.015f,27),line,0);
    box(v3(0,-0.005f,0),v3(17.8f,0.015f,0.10f),line,0);
    box(v3(0,0.15f,-25.5f),v3(4.0f,0.15f,0.25f),v3(0.85f,0.25f,0.12f),0);
    box(v3(0,0.15f,25.5f),v3(4.0f,0.15f,0.25f),v3(0.12f,0.45f,0.9f),0);
}

static void build_car(void)
{
    Vec3 body=v3(0.95f,0.16f,0.12f), glass=v3(0.12f,0.22f,0.34f), tire=v3(0.035f,0.04f,0.05f);
    Vec3 forward=v3(sinf(car_yaw),0,cosf(car_yaw));
    Vec3 pos=v3(car_x,0.20f+car_y,car_z);
    box(add(pos,mul(forward,0.10f)),v3(0.95f,0.28f,1.45f),body,car_yaw);
    box(add(pos,mul(forward,-0.20f)),v3(0.72f,0.38f,0.72f),glass,car_yaw);
    for(int side=-1;side<=1;side+=2) for(int axle=-1;axle<=1;axle+=2) {
        Vec3 right=v3(cosf(car_yaw),0,-sinf(car_yaw));
        Vec3 wheel=add(add(pos,mul(right,(float)side*0.88f)),mul(forward,(float)axle*0.88f));
        box(add(wheel,v3(0,-0.16f,0)),v3(0.18f,0.25f,0.36f),tire,car_yaw);
    }
}

static void build_scene(void)
{
    vertex_count=0;
    build_arena();
    build_car();
    sphere(v3(ball_x,ball_y,ball_z),1.0f,v3(0.92f,0.93f,0.96f));
}

static void write_vertices(void)
{
    const uint32_t bytes=vertex_count*sizeof(Vertex);
    const uint32_t *source=(const uint32_t *)vertices;
    for(uint32_t i=0;i<bytes/4;i++) gpu_vertices[i]=source[i];
    xsdk_gpu_invalidate(xsdk_gpu_address(gpu_vertices),bytes);
}

static void update_game(const INPUT_STATE *input)
{
    float steer=stick(input->gamepad.thumb_lx);
    float camera=stick(input->gamepad.thumb_rx);
    float camera_tilt=stick(input->gamepad.thumb_ry);
    camera_yaw-=camera*0.045f;
    camera_pitch=clampf(camera_pitch+camera_tilt*0.025f,-0.12f,0.85f);

    float throttle=(float)input->gamepad.right_trigger/255.0f;
    float brake=(float)input->gamepad.left_trigger/255.0f;
    float max_speed=(input->gamepad.buttons&INPUT_GAMEPAD_X)?0.32f:0.22f;
    car_speed+=throttle*0.010f;
    car_speed-=brake*0.012f;
    if(throttle<0.02f&&brake<0.02f) car_speed*=0.985f;
    car_speed=clampf(car_speed,-0.12f,max_speed);
    if(fabsf(car_speed)>0.003f) car_yaw+=steer*0.045f*(car_speed>=0?1.0f:-1.0f);
    car_x+=sinf(car_yaw)*car_speed;
    car_z+=cosf(car_yaw)*car_speed;
    car_x=clampf(car_x,-15.5f,15.5f);
    car_z=clampf(car_z,-25.0f,25.0f);

    if((input->gamepad.buttons&INPUT_GAMEPAD_A)&&car_y<=0.001f) car_y=0.55f;
    car_y-=0.018f;
    if(car_y<0) car_y=0;

    ball_vy-=0.0045f;
    ball_x+=ball_vx; ball_y+=ball_vy; ball_z+=ball_vz;
    if(ball_y<1.0f) { ball_y=1.0f; ball_vy=fabsf(ball_vy)*0.72f; }
    ball_vx*=0.995f; ball_vz*=0.995f; ball_vy*=0.995f;
    float dx=ball_x-car_x, dz=ball_z-car_z;
    if(dx*dx+dz*dz<3.1f&&ball_y<2.2f) {
        float push=car_speed*0.8f+0.08f;
        ball_vx+=sinf(car_yaw)*push;
        ball_vz+=cosf(car_yaw)*push;
        ball_vy+=0.035f;
    }
    if(ball_x>16.8f||ball_x< -16.8f) ball_vx=-ball_vx*0.7f;
    if(ball_z>26.8f||ball_z< -26.8f) ball_vz=-ball_vz*0.7f;
}

int main(void)
{
    xsdk_log_write("Rocket-X: starting 3D prototype");
    NTSTATUS status=xsdk_gpu_open();
    if(!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: GPU init failed %08X",(unsigned)status); return 1; }

    status=xsdk_shader_load(&vertex_shader,rocket_vert.container,rocket_vert.container_bytes);
    if(NT_SUCCESS(status)) status=xsdk_shader_load(&pixel_shader,rocket_frag.container,rocket_frag.container_bytes);
    if(!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: shader load failed %08X",(unsigned)status); xsdk_gpu_close(); return 1; }

    const XsdkVertexElement layout[]={
        {.usage=XSDK_USAGE_POSITION,.format=XSDK_VERTEX_FLOAT3,.offset_bytes=0},
        {.usage=XSDK_USAGE_COLOR,.format=XSDK_VERTEX_FLOAT3,.offset_bytes=12}
    };
    status=xsdk_shader_set_vertex_layout(&vertex_shader,layout,2,(const uint32_t[]){VERTEX_BYTES});
    if(!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: vertex layout failed %08X",(unsigned)status); return 1; }

    const XsdkDepthStencil depth={.depth_test=true,.depth_write=true,.depth_compare=XSDK_COMPARE_LESS_EQUAL};
    xsdk_gpu_depth_stencil_state(&depth_state,&depth);
    gpu_vertices=xsdk_gpu_alloc(MAX_VERTICES*VERTEX_BYTES);
    if(!gpu_vertices) { xsdk_log_write("Rocket-X: GPU vertex allocation failed"); return 1; }

    uint32_t width=0,height=0;
    status=xsdk_present_open(XSDK_DISPLAY_720P,BUFFERS,SWAP_INTERVAL,&width,&height);
    if(!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: display init failed %08X",(unsigned)status); return 1; }

    xsdk_log_write("Rocket-X: left stick steer, RT accelerate, LT brake, A jump, X boost, right stick camera, Back exit");
    for(;;) {
        INPUT_STATE input={0};
        if(XamInputGetState(0,0,&input)==ERROR_SUCCESS&&(input.gamepad.buttons&INPUT_GAMEPAD_BACK)) break;
        update_game(&input);
        build_scene();
        write_vertices();

        uint32_t address=0;
        status=xsdk_present_begin(&address,BEGIN_TIMEOUT_MS);
        if(!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: present begin failed %08X",(unsigned)status); break; }
        xsdk_gpu_set_target(width,height);
        xsdk_gpu_set_depth_stencil(&depth_state);
        xsdk_gpu_set_shaders(&vertex_shader,&pixel_shader);
        xsdk_built_shader_set_literals(&rocket_vert);
        xsdk_built_shader_set_literals(&rocket_frag);
        Mat4 mvp=camera_matrix((float)width/(float)height);
        xsdk_gpu_set_vertex_constants(ROCKET_VERT_PUSH,mvp.m,ROCKET_VERT_PUSH_REGISTERS);
        xsdk_gpu_set_vertices(0,xsdk_gpu_address(gpu_vertices),vertex_count*VERTEX_BYTES);
        xsdk_gpu_draw(XSDK_PRIMITIVE_TRIANGLE_LIST,vertex_count);
        xsdk_gpu_resolve(address,width,height,CLEAR_ARGB);
        xsdk_present_end();
    }

    xsdk_present_close();
    xsdk_gpu_free(gpu_vertices);
    xsdk_shader_free(&pixel_shader);
    xsdk_shader_free(&vertex_shader);
    status=xsdk_gpu_close();
    xsdk_log_write("Rocket-X: exit status %08X",(unsigned)status);
    return 0;
}
