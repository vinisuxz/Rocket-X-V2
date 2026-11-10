#include "game.h"

#include <math.h>

#define PI_F 3.14159265f

static const float GRAVITY = 6.5f;
static const float STICKY = 9.0f;               // empurra o carro contra a superficie em que anda
static const float MAX_DRIVE_SPEED = 14.1f;
static const float MAX_SPEED = 23.0f;
static const float DRIVE_ACCEL = 16.0f;
static const float BRAKE_ACCEL = 35.0f;
static const float COAST_DECEL = 4.5f;
static const float BOOST_ACCEL = 10.5f;
static const float BOOST_USE = 33.3f;           // por segundo
static const float JUMP_SPEED = 3.0f;
static const float JUMP_HOLD_ACCEL = 14.6f;
static const float JUMP_HOLD_TIME = 0.2f;
static const float DODGE_SPEED = 5.0f;
static const float DODGE_WINDOW = 1.25f;
static const float FLIP_TIME = 0.6f;            // uma volta inteira
static const float MAX_ANGULAR = 5.5f;
static const float BALL_MAX_SPEED = 60.0f;
static const float BALL_BOUNCE = 0.6f;
static const float SUBSTEP = 1.0f / 120.0f;

// Contornos do carro, no referencial dele (esquerda, cima, frente) a partir do centro do casco.
static const float HULL_LEFT = 0.39f, HULL_FRONT = 0.76f, HULL_TOP = 0.55f, HULL_BOTTOM = -0.10f;
static const float ROOF_LEFT = 0.30f, ROOF_FRONT = 0.40f, ROOF_UP = 0.50f;
static const float WHEEL_LEFT = 0.40f, WHEEL_FRONT = 0.46f;

static const float CAMERA_DISTANCE_NEAR = 3.0f;
static const float CAMERA_DISTANCE_FAR = 4.2f;
static const float CAMERA_DEFAULT_PITCH = 0.20f;
static const float CAMERA_ORBIT_SPEED = 2.6f;
static const float CAMERA_TILT_SPEED = 1.6f;
static const float CAMERA_RETURN_DELAY = 1.0f;
static const float CAMERA_FOLLOW = 4.0f;

static float wrap_pi(float a)
{
    while (a > PI_F) a -= 2.0f * PI_F;
    while (a < -PI_F) a += 2.0f * PI_F;
    return a;
}

// ---- a arena ---------------------------------------------------------------------------------------------------

// Distancia assinada ao "miolo" da arena: o campo encolhido pelo raio das rampas. A arena e esse miolo engordado
// por ARENA_FILLET_RADIUS, o que da as rampas curvas de graca.
static float core_distance(Vec3 p)
{
    const float r = ARENA_FILLET_RADIUS;
    const float rc = ARENA_CORNER_RADIUS - r;
    float qx = fabsf(p.x) - (ARENA_HALF_X - r - rc);
    float qz = fabsf(p.z) - (ARENA_HALF_Z - r - rc);
    float ox = qx > 0.0f ? qx : 0.0f, oz = qz > 0.0f ? qz : 0.0f;
    float plan = sqrtf(ox * ox + oz * oz) + fminf(fmaxf(qx, qz), 0.0f) - rc;
    float qy = fabsf(p.y - ARENA_HEIGHT * 0.5f) - (ARENA_HEIGHT * 0.5f - r);
    float oy = qy > 0.0f ? qy : 0.0f;
    float po = plan > 0.0f ? plan : 0.0f;
    return sqrtf(po * po + oy * oy) + fminf(fmaxf(plan, qy), 0.0f);
}

Surface arena_surface(Vec3 p)
{
    const float e = 0.02f;
    Vec3 g = v3(core_distance(v3(p.x + e, p.y, p.z)) - core_distance(v3(p.x - e, p.y, p.z)),
                core_distance(v3(p.x, p.y + e, p.z)) - core_distance(v3(p.x, p.y - e, p.z)),
                core_distance(v3(p.x, p.y, p.z + e)) - core_distance(v3(p.x, p.y, p.z - e)));
    Surface s;
    s.distance = ARENA_FILLET_RADIUS - core_distance(p);
    s.normal = norm(mul(g, -1.0f));
    if (length(s.normal) < 0.5f) s.normal = v3(0, 1, 0);   // no eixo do miolo; qualquer um serve
    return s;
}

// ---- o carro ---------------------------------------------------------------------------------------------------

static Vec3 car_left(const Car *car) { return cross(car->up, car->fwd); }
static Vec3 car_right(const Car *car) { return cross(car->fwd, car->up); }

static void orthonormalize(Car *car)
{
    car->fwd = norm(sub(car->fwd, mul(car->up, dot(car->fwd, car->up))));
    if (length(car->fwd) < 0.5f) car->fwd = v3(1, 0, 0);
    car->up = norm(cross(car_right(car), car->fwd));
    car->up = norm(car->up);
    car->fwd = norm(sub(car->fwd, mul(car->up, dot(car->fwd, car->up))));
}

static void rotate_car(Car *car, Vec3 axis_times_angle)
{
    float angle = length(axis_times_angle);
    if (angle < 1e-6f) return;
    Vec3 axis = mul(axis_times_angle, 1.0f / angle);
    car->fwd = rotate_about(car->fwd, axis, angle);
    car->up = rotate_about(car->up, axis, angle);
    orthonormalize(car);
}

static Vec3 car_point(const Car *car, float left, float up, float front)
{
    return add(car->pos, add(mul(car_left(car), left), add(mul(car->up, up), mul(car->fwd, front))));
}

void game_car_reset(Car *car, Vec3 pos, Vec3 fwd)
{
    *car = (Car){0};
    car->pos = pos;
    car->fwd = fwd;
    car->up = v3(0, 1, 0);
    car->ground_normal = car->up;
    car->boost = 33.0f;
    car->on_ground = true;
}

Mat4 game_car_matrix(const Car *car)
{
    Vec3 left = car_left(car);
    Vec3 origin = car->pos;
    Mat4 m = mat4_identity();
    m.m[0] = left.x;    m.m[1] = left.y;    m.m[2] = left.z;
    m.m[4] = car->up.x; m.m[5] = car->up.y; m.m[6] = car->up.z;
    m.m[8] = car->fwd.x; m.m[9] = car->fwd.y; m.m[10] = car->fwd.z;
    m.m[12] = origin.x; m.m[13] = origin.y; m.m[14] = origin.z;
    return m;
}

// Testa as 4 rodas e os 4 cantos do teto; empurra o carro para fora da arena e devolve o que tocou.
static void car_collide(Car *car)
{
    Vec3 wheel_normal = v3(0, 0, 0), roof_normal = v3(0, 0, 0);
    int wheels = 0, roof = 0;
    for (int i = 0; i < 8; i++) {
        bool is_wheel = i < 4;
        float l = (i & 1) ? WHEEL_LEFT : -WHEEL_LEFT;
        float f = (i & 2) ? WHEEL_FRONT : -WHEEL_FRONT;
        float u = is_wheel ? -CAR_HULL_CENTER : ROOF_UP;
        if (!is_wheel) { l = (i & 1) ? ROOF_LEFT : -ROOF_LEFT; f = (i & 2) ? ROOF_FRONT : -ROOF_FRONT; }
        Surface s = arena_surface(car_point(car, l, u, f));
        if (s.distance > (is_wheel ? 0.06f : 0.0f)) continue;
        if (is_wheel) { wheel_normal = add(wheel_normal, s.normal); wheels++; }
        else { roof_normal = add(roof_normal, s.normal); roof++; }
        if (s.distance < 0.0f) {
            car->pos = add(car->pos, mul(s.normal, -s.distance));
            float into = dot(car->vel, s.normal);
            if (into < 0.0f) car->vel = sub(car->vel, mul(s.normal, into * (is_wheel ? 1.0f : 1.3f)));
        }
    }
    car->touching = wheels + roof > 0;
    car->on_ground = wheels >= 2;
    if (wheels > 0) car->ground_normal = norm(wheel_normal);
    else if (roof > 0) car->ground_normal = norm(roof_normal);
}

static void car_ground(Car *car, const Controls *c, float dt)
{
    // Alinha o carro a superficie (o que o faz subir pela parede).
    float k = 1.0f - expf(-14.0f * dt);
    car->up = norm(add(car->up, mul(sub(car->ground_normal, car->up), k)));
    orthonormalize(car);
    car->ang = v3(0, 0, 0);

    Vec3 n = car->up;
    float speed_n = dot(car->vel, n);
    Vec3 tangent = sub(car->vel, mul(n, speed_n));
    float forward = dot(tangent, car->fwd);
    Vec3 right = car_right(car);
    float lateral = dot(tangent, right);

    float drive = c->throttle - c->brake;
    float accel;
    if (fabsf(drive) < 0.02f) accel = -(forward > 0 ? 1.0f : -1.0f) * fminf(COAST_DECEL, fabsf(forward) / dt);
    else if (drive * forward < -0.5f) accel = drive * BRAKE_ACCEL;
    else accel = drive * DRIVE_ACCEL * fmaxf(0.0f, 1.0f - fabsf(forward) / MAX_DRIVE_SPEED);
    bool boosting = c->boost && car->boost > 0.0f;
    if (boosting) {
        accel += BOOST_ACCEL;
        car->boost = fmaxf(0.0f, car->boost - BOOST_USE * dt);
    }
    car->boosting = boosting;
    forward += accel * dt;
    forward = clampf(forward, -MAX_DRIVE_SPEED, MAX_SPEED);
    if (!boosting && forward > MAX_DRIVE_SPEED) forward = fmaxf(MAX_DRIVE_SPEED, forward - 8.0f * dt);

    lateral *= expf(-(c->handbrake ? 1.5f : 14.0f) * dt);

    // Esterco: vira rapido devagar e menos em alta velocidade; so gira se estiver andando.
    float speed = fabsf(forward);
    float turn_rate = lerpf(2.4f, 1.0f, clampf(speed / MAX_DRIVE_SPEED, 0.0f, 1.0f));
    if (c->handbrake) turn_rate *= 1.6f;
    float moving = clampf(speed / 1.5f, 0.0f, 1.0f);
    float yaw = -c->steer_x * turn_rate * moving * (forward >= 0 ? 1.0f : -1.0f);
    Vec3 old_fwd = car->fwd;
    car->fwd = rotate_about(car->fwd, car->up, yaw * dt);
    orthonormalize(car);
    // A velocidade acompanha o nariz (menos a parte lateral, que e a derrapagem).
    (void)old_fwd;
    car->vel = add(add(mul(car->fwd, forward), mul(car_right(car), lateral)), mul(n, speed_n));
    car->wheel_distance += forward * dt;
    car->steer_visual = lerpf(car->steer_visual, -c->steer_x * 0.5f, 1.0f - expf(-14.0f * dt));
    car->vel = sub(car->vel, mul(car->ground_normal, STICKY * dt));
}

static void car_air(Car *car, const Controls *c, float dt)
{
    car->steer_visual = lerpf(car->steer_visual, 0.0f, 1.0f - expf(-8.0f * dt));
    bool boosting = c->boost && car->boost > 0.0f;
    car->boosting = boosting;
    if (boosting) {
        car->vel = add(car->vel, mul(car->fwd, BOOST_ACCEL * dt));
        car->boost = fmaxf(0.0f, car->boost - BOOST_USE * dt);
        if (length(car->vel) > MAX_SPEED) car->vel = mul(norm(car->vel), MAX_SPEED);
    }

    if (car->flip_time > 0.0f) {
        // Mortal em andamento: gira nos eixos do carro, sem o jogador.
        car->ang = add(add(mul(car_left(car), car->flip_rate.x), mul(car->up, car->flip_rate.y)),
                       mul(car->fwd, car->flip_rate.z));
        car->flip_time -= dt;
        if (car->flip_time <= 0.0f) car->ang = mul(car->ang, 0.2f);
    } else {
        // Ar: pitch com o analogico para frente/tras, guinada para os lados, rolagem nos botoes.
        float roll = (c->roll_right ? 1.0f : 0.0f) - (c->roll_left ? 1.0f : 0.0f);
        float input[3] = {c->steer_y, -c->steer_x, roll};       // esquerda, cima, frente
        const float torque[3] = {12.0f, 9.0f, 30.0f};
        const float damp[3] = {2.8f, 1.9f, 4.1f};
        Vec3 axis[3] = {car_left(car), car->up, car->fwd};
        Vec3 next = v3(0, 0, 0);
        for (int i = 0; i < 3; i++) {
            float v = dot(car->ang, axis[i]);
            v += input[i] * torque[i] * dt;
            v -= v * damp[i] * (1.0f - fminf(1.0f, fabsf(input[i]))) * dt;
            next = add(next, mul(axis[i], v));
        }
        float a = length(next);
        car->ang = a > MAX_ANGULAR ? mul(next, MAX_ANGULAR / a) : next;
    }
    rotate_car(car, mul(car->ang, dt));
}

static void car_jump(Car *car, const Controls *c, float dt)
{
    bool pressed = c->jump && !car->jump_was_down;
    car->jump_was_down = c->jump;
    car->air_lock = fmaxf(0.0f, car->air_lock - dt);
    car->dodge_window = fmaxf(0.0f, car->dodge_window - dt);

    if (pressed && car->touching && car->air_lock <= 0.0f) {
        Vec3 n = car->on_ground ? car->up : car->ground_normal;
        car->vel = add(car->vel, mul(n, JUMP_SPEED));
        car->jump_time = JUMP_HOLD_TIME;
        car->air_lock = 0.12f;
        car->can_dodge = true;
        car->dodge_window = DODGE_WINDOW;
        car->on_ground = false;
        car->touching = false;
    } else if (pressed && !car->touching && car->can_dodge && car->dodge_window > 0.0f && car->flip_time <= 0.0f) {
        car->can_dodge = false;
        car->jump_time = 0.0f;
        float sx = c->steer_x, sy = c->steer_y;
        float mag = sqrtf(sx * sx + sy * sy);
        if (mag > 0.3f) {
            // Impulso para onde o analogico aponta e uma volta completa em torno do eixo perpendicular.
            Vec3 push = add(mul(car->fwd, sy), mul(car_right(car), sx));
            car->vel = add(car->vel, mul(norm(push), DODGE_SPEED));
            float rate = 2.0f * PI_F / FLIP_TIME;
            car->flip_rate = v3(sy / mag * rate, 0.0f, sx / mag * rate);
            car->flip_time = FLIP_TIME;
        } else {
            car->vel = add(car->vel, mul(car->up, JUMP_SPEED));
        }
    }
    if (c->jump && car->jump_time > 0.0f) {
        car->vel = add(car->vel, mul(car->up, JUMP_HOLD_ACCEL * dt));
        car->jump_time -= dt;
    } else if (!c->jump) {
        car->jump_time = 0.0f;
    }
}

static void car_step(Car *car, const Controls *c, float dt)
{
    car_jump(car, c, dt);
    if (car->air_lock > 0.0f) { car->on_ground = false; }
    if (car->on_ground) car_ground(car, c, dt);
    else car_air(car, c, dt);

    if (!car->on_ground) car->vel.y -= GRAVITY * dt;
    else car->vel.y -= GRAVITY * dt;       // a gravidade vale em toda parte; o esterco segura o carro na parede
    car->pos = add(car->pos, mul(car->vel, dt));

    bool locked = car->air_lock > 0.0f;
    car_collide(car);
    if (locked) car->on_ground = false;
    if (car->on_ground) {
        car->can_dodge = false;
        car->flip_time = 0.0f;
        car->dodge_window = 0.0f;
    }
}

// ---- a bola ----------------------------------------------------------------------------------------------------

static void ball_step(Game *game, float dt)
{
    Ball *ball = &game->ball;
    const Car *car = &game->car;
    ball->vel.y -= GRAVITY * dt;
    ball->vel = mul(ball->vel, expf(-0.03f * dt));
    ball->pos = add(ball->pos, mul(ball->vel, dt));

    Surface s = arena_surface(ball->pos);
    if (s.distance < BALL_RADIUS) {
        ball->pos = add(ball->pos, mul(s.normal, BALL_RADIUS - s.distance));
        float into = dot(ball->vel, s.normal);
        if (into < 0.0f) {
            float bounce = into < -1.5f ? BALL_BOUNCE : 0.0f;
            ball->vel = sub(ball->vel, mul(s.normal, (1.0f + bounce) * into));
        }
        Vec3 n = s.normal;
        Vec3 tangent = sub(ball->vel, mul(n, dot(ball->vel, n)));
        ball->vel = sub(ball->vel, mul(tangent, 0.35f * dt));       // atrito de rolagem
    }

    // Bola contra o casco do carro, uma caixa no referencial dele.
    Vec3 rel = sub(ball->pos, car->pos);
    Vec3 local = v3(dot(rel, car_left(car)), dot(rel, car->up), dot(rel, car->fwd));
    Vec3 closest = v3(clampf(local.x, -HULL_LEFT, HULL_LEFT), clampf(local.y, HULL_BOTTOM, HULL_TOP),
                      clampf(local.z, -HULL_FRONT, HULL_FRONT));
    Vec3 delta = sub(local, closest);
    float d = length(delta);
    if (d < BALL_RADIUS) {
        Vec3 n_local = d > 1e-4f ? mul(delta, 1.0f / d) : v3(0, 1, 0);
        Vec3 n = add(add(mul(car_left(car), n_local.x), mul(car->up, n_local.y)), mul(car->fwd, n_local.z));
        ball->pos = add(ball->pos, mul(n, BALL_RADIUS - d));
        Vec3 car_vel = car->vel;
        float vn = dot(sub(ball->vel, car_vel), n);
        if (vn < 0.0f) {
            float speed = length(car->vel);
            ball->vel = add(ball->vel, mul(n, -(1.0f + 0.55f) * vn + speed * 0.25f));
            // O toque levanta a bola um pouco, como no jogo de verdade.
            ball->vel.y += fabsf(vn) * 0.15f;
        }
    }
    float speed = length(ball->vel);
    if (speed > BALL_MAX_SPEED) ball->vel = mul(ball->vel, BALL_MAX_SPEED / speed);
}

// ---- a camera --------------------------------------------------------------------------------------------------

static void camera_update(Game *game, const Controls *c, float dt)
{
    Camera *cam = &game->camera;
    const Car *car = &game->car;
    bool active = fabsf(c->look_x) > 0.01f || fabsf(c->look_y) > 0.01f;
    cam->yaw = wrap_pi(cam->yaw - c->look_x * CAMERA_ORBIT_SPEED * dt);
    cam->pitch = clampf(cam->pitch - c->look_y * CAMERA_TILT_SPEED * dt, 0.03f, 1.3f);
    cam->idle = active ? 0.0f : cam->idle + dt;
    if (cam->idle > CAMERA_RETURN_DELAY) {
        float blend = 1.0f - expf(-CAMERA_FOLLOW * dt);
        float horizontal = sqrtf(car->fwd.x * car->fwd.x + car->fwd.z * car->fwd.z);
        if (horizontal > 0.35f) {
            float car_yaw = atan2f(car->fwd.x, car->fwd.z);
            cam->yaw = wrap_pi(cam->yaw + wrap_pi(car_yaw - cam->yaw) * blend);
        }
        cam->pitch += (CAMERA_DEFAULT_PITCH - cam->pitch) * blend;
    }

    Vec3 goal = add(car->pos, mul(car->up, 0.35f));
    cam->target = add(cam->target, mul(sub(goal, cam->target), 1.0f - expf(-18.0f * dt)));
    cam->distance = lerpf(CAMERA_DISTANCE_NEAR, CAMERA_DISTANCE_FAR, clampf(length(car->vel) / MAX_SPEED, 0.0f, 1.0f));

    float cp = cosf(cam->pitch);
    Vec3 back = v3(-sinf(cam->yaw) * cp, sinf(cam->pitch), -cosf(cam->yaw) * cp);
    cam->eye = add(cam->target, mul(back, cam->distance));
    // A camera nao atravessa a arena.
    for (int i = 0; i < 3; i++) {
        Surface s = arena_surface(cam->eye);
        if (s.distance < 0.5f) cam->eye = add(cam->eye, mul(s.normal, 0.5f - s.distance));
    }
}

// ---- o jogo ----------------------------------------------------------------------------------------------------

void game_reset(Game *game)
{
    game_car_reset(&game->car, v3(-30.0f, CAR_HULL_CENTER + 0.01f, 0.0f), v3(1, 0, 0));
    game->ball = (Ball){0};
    game->ball.pos = v3(0.0f, BALL_RADIUS, 0.0f);
}

void game_init(Game *game)
{
    game_reset(game);
    game->camera = (Camera){0};
    game->camera.yaw = atan2f(game->car.fwd.x, game->car.fwd.z);
    game->camera.pitch = CAMERA_DEFAULT_PITCH;
    game->camera.distance = CAMERA_DISTANCE_NEAR;
    game->camera.target = add(game->car.pos, v3(0.0f, 0.35f, 0.0f));
    camera_update(game, &(Controls){0}, 0.0f);
}

void game_update(Game *game, const Controls *controls, float dt)
{
    if (!(dt > 0.0f)) dt = 0.0f;
    dt = fminf(dt, 1.0f / 15.0f);
    if (controls->reset) game_reset(game);

    int steps = (int)ceilf(dt / SUBSTEP);
    if (steps < 1) steps = 1;
    float h = dt / (float)steps;
    for (int i = 0; i < steps && dt > 0.0f; i++) {
        car_step(&game->car, controls, h);
        ball_step(game, h);
    }
    camera_update(game, controls, dt);
}

Mat4 game_view(const Game *game)
{
    return mat4_look_at(game->camera.eye, game->camera.target);
}
