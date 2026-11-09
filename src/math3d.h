#ifndef ROCKETX_MATH3D_H
#define ROCKETX_MATH3D_H

#include <math.h>

// O mundo e destro, com Y para cima. Matrizes ficam por coluna: m[coluna * 4 + linha].
typedef struct { float x, y, z; } Vec3;
typedef struct { float m[16]; } Mat4;

static inline Vec3 v3(float x, float y, float z) { return (Vec3){x, y, z}; }
static inline Vec3 add(Vec3 a, Vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline Vec3 sub(Vec3 a, Vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline Vec3 mul(Vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline Vec3 cross(Vec3 a, Vec3 b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static inline float length(Vec3 a) { return sqrtf(dot(a, a)); }
static inline Vec3 norm(Vec3 a) { float n = length(a); return n > 0.0001f ? mul(a, 1.0f / n) : v3(0, 0, 0); }
static inline float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }

static inline Mat4 mat4_identity(void)
{
    return (Mat4){{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
}

// a * b: aplica b primeiro, depois a.
static inline Mat4 mat4_multiply(const Mat4 *a, const Mat4 *b)
{
    Mat4 r;
    for (int c = 0; c < 4; c++) {
        for (int row = 0; row < 4; row++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += a->m[k * 4 + row] * b->m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    }
    return r;
}

static inline Mat4 mat4_translation(Vec3 t)
{
    Mat4 r = mat4_identity();
    r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z;
    return r;
}

// Gira em Y: +Z vai para (sen, 0, cos).
static inline Mat4 mat4_rotation_y(float angle)
{
    Mat4 r = mat4_identity();
    float c = cosf(angle), s = sinf(angle);
    r.m[0] = c; r.m[2] = -s; r.m[8] = s; r.m[10] = c;
    return r;
}

// Gira em X: o topo de uma roda que gira para a frente (+Z) tem angulo positivo.
static inline Mat4 mat4_rotation_x(float angle)
{
    Mat4 r = mat4_identity();
    float c = cosf(angle), s = sinf(angle);
    r.m[5] = c; r.m[6] = s; r.m[9] = -s; r.m[10] = c;
    return r;
}

static inline Vec3 mat4_transform_point(const Mat4 *m, Vec3 p)
{
    return v3(m->m[0] * p.x + m->m[4] * p.y + m->m[8] * p.z + m->m[12],
              m->m[1] * p.x + m->m[5] * p.y + m->m[9] * p.z + m->m[13],
              m->m[2] * p.x + m->m[6] * p.y + m->m[10] * p.z + m->m[14]);
}

static inline Mat4 mat4_look_at(Vec3 eye, Vec3 target)
{
    Vec3 f = norm(sub(target, eye));
    Vec3 s = norm(cross(f, v3(0, 1, 0)));   // direita da camera
    Vec3 u = cross(s, f);
    Mat4 view = mat4_identity();
    view.m[0] = s.x; view.m[4] = s.y; view.m[8] = s.z; view.m[12] = -dot(s, eye);
    view.m[1] = u.x; view.m[5] = u.y; view.m[9] = u.z; view.m[13] = -dot(u, eye);
    view.m[2] = -f.x; view.m[6] = -f.y; view.m[10] = -f.z; view.m[14] = dot(f, eye);
    return view;
}

// Profundidade de 0 (plano proximo) a 1 (plano distante).
static inline Mat4 mat4_perspective(float fov, float aspect, float near_z, float far_z)
{
    float fs = 1.0f / tanf(fov * 0.5f);
    float fz = far_z / (near_z - far_z);
    return (Mat4){{fs / aspect, 0, 0, 0, 0, fs, 0, 0, 0, 0, fz, -1, 0, 0, near_z * fz, 0}};
}

#endif
