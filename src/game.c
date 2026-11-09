#include "game.h"

#include <math.h>

// Valores no estilo do Rocket League (la 1 uu = 1 cm, aqui 100 uu = 1 m).
static const float GRAVITY = 6.5f;
static const float THROTTLE_ACCEL = 16.0f;
static const float BRAKE_DECEL = 35.0f;
static const float COAST_DECEL = 5.25f;
static const float BOOST_ACCEL = 9.92f;
static const float MAX_DRIVE_SPEED = 14.1f;
static const float MAX_SPEED = 23.0f;
static const float BOOST_USE = 33.3f;
static const float BOOST_REGEN = 10.0f;
static const float JUMP_SPEED = 3.0f;
static const float JUMP_HOLD_ACCEL = 14.6f;
static const float JUMP_HOLD_TIME = 0.2f;
static const float STEER_RATE_SLOW = 2.6f;
static const float STEER_RATE_FAST = 0.5f;
static const float GRIP = 14.0f;
static const float GRIP_HANDBRAKE = 1.2f;
static const float AIR_PITCH_RATE = 5.0f;
static const float AIR_YAW_RATE = 2.5f;

static const float CAR_RADIUS = 0.75f;          // o carro bate nas paredes como um circulo
static const float CAR_TOP = 0.66f;
static const float CAR_HALF_WIDTH = 0.38f;
static const float CAR_HALF_HEIGHT = 0.28f;
static const float CAR_HALF_LENGTH = 0.72f;
static const float CAR_CENTER_HEIGHT = 0.36f;
static const float CAR_MASS = 180.0f;
static const float BALL_MASS = 30.0f;
static const float BALL_BOUNCE = 0.6f;
static const float BALL_DRAG = 0.03f;
static const float BALL_ROLL_DRAG = 0.25f;
static const float BALL_MAX_SPEED = 60.0f;
static const float HIT_RESTITUTION = 0.55f;
static const float HIT_EXTRA = 0.35f;

static const float CAMERA_ORBIT_SPEED = 3.0f;
static const float CAMERA_TILT_SPEED = 1.6f;
static const float CAMERA_RETURN_DELAY = 0.8f;
static const float CAMERA_FOLLOW = 3.5f;
static const float CAMERA_DEFAULT_PITCH = 0.30f;
static const float CAMERA_DISTANCE_NEAR = 4.6f;
static const float CAMERA_DISTANCE_FAR = 6.6f;

static const float PI_F = 3.14159265358979f;
static const float MAX_STEP = 1.0f / 120.0f;

static float wrap_pi(float angle)
{
    while (angle > PI_F) angle -= 2.0f * PI_F;
    while (angle < -PI_F) angle += 2.0f * PI_F;
    return angle;
}

static Vec3 car_forward(const Car *car)
{
    return v3(sinf(car->yaw) * cosf(car->pitch), sinf(car->pitch), cosf(car->yaw) * cosf(car->pitch));
}

// Para dentro do retangulo arredondado do campo, para um corpo de `radius` no plano XZ. Se bateu, tira a parte da
// velocidade que vai contra a parede (com `restitution` de volta) e diz que bateu.
static bool arena_push_xz(Vec3 *pos, Vec3 *vel, float radius, float restitution)
{
    float half_x = ARENA_HALF_X - radius, half_z = ARENA_HALF_Z - radius, corner = ARENA_CORNER_RADIUS - radius;
    float qx = fabsf(pos->x) - (half_x - corner), qz = fabsf(pos->z) - (half_z - corner);
    float nx, nz, outside;
    if (qx > 0.0f && qz > 0.0f) {
        float len = sqrtf(qx * qx + qz * qz);
        outside = len - corner;
        nx = qx / len;
        nz = qz / len;
    } else if (qx > qz) {
        outside = qx - corner; nx = 1.0f; nz = 0.0f;
    } else {
        outside = qz - corner; nx = 0.0f; nz = 1.0f;
    }
    if (outside <= 0.0f) return false;
    nx = pos->x < 0.0f ? -nx : nx;
    nz = pos->z < 0.0f ? -nz : nz;
    pos->x -= nx * outside;
    pos->z -= nz * outside;
    float into = vel->x * nx + vel->z * nz;
    if (into > 0.0f) {
        vel->x -= (1.0f + restitution) * into * nx;
        vel->z -= (1.0f + restitution) * into * nz;
    }
    return true;
}

static void car_step(Car *car, const Controls *c, float dt)
{
    // Pulo: um impulso ao apertar e um empurrao a mais enquanto segura.
    if (c->jump && !car->jump_was_down && car->on_ground) {
        car->vel.y = JUMP_SPEED;
        car->jump_time = JUMP_HOLD_TIME;
        car->on_ground = false;
    }
    if (c->jump && car->jump_time > 0.0f) {
        car->vel.y += JUMP_HOLD_ACCEL * dt;
        car->jump_time -= dt;
    } else {
        car->jump_time = 0.0f;
    }
    car->jump_was_down = c->jump;

    car->boosting = c->boost && car->boost > 0.0f;
    if (car->boosting) car->boost = fmaxf(0.0f, car->boost - BOOST_USE * dt);
    else if (car->on_ground) car->boost = fminf(CAR_MAX_BOOST, car->boost + BOOST_REGEN * dt);

    float steer = clampf(c->steer_x, -1.0f, 1.0f);
    car->steer_visual += (-steer * 0.5f - car->steer_visual) * fminf(1.0f, 12.0f * dt);

    if (car->on_ground) {
        Vec3 fwd = v3(sinf(car->yaw), 0.0f, cosf(car->yaw));
        float vf_before = dot(car->vel, fwd);
        float speed_ratio = clampf(fabsf(vf_before) / MAX_SPEED, 0.0f, 1.0f);

        // Esterca: vira a frente. Nao vira parado, e de re o sentido troca.
        float rate = lerpf(STEER_RATE_SLOW, STEER_RATE_FAST, speed_ratio);
        if (c->handbrake) rate *= 1.8f;
        float moving = clampf(fabsf(vf_before) / 1.5f, 0.0f, 1.0f);
        car->yaw = wrap_pi(car->yaw - steer * rate * moving * (vf_before >= -0.1f ? 1.0f : -1.0f) * dt);

        // Com a frente nova, separa a velocidade em frente e de lado.
        fwd = v3(sinf(car->yaw), 0.0f, cosf(car->yaw));
        Vec3 right = v3(-cosf(car->yaw), 0.0f, sinf(car->yaw));
        float vf = dot(car->vel, fwd), vl = dot(car->vel, right);

        float accel = car->boosting ? BOOST_ACCEL : 0.0f;
        if (c->throttle > 0.01f) {
            if (vf < -0.1f) accel += BRAKE_DECEL * c->throttle;
            else accel += THROTTLE_ACCEL * c->throttle * clampf(1.0f - vf / MAX_DRIVE_SPEED, 0.0f, 1.0f);
        }
        if (c->brake > 0.01f) {
            if (vf > 0.5f) accel -= BRAKE_DECEL * c->brake;
            else accel -= THROTTLE_ACCEL * c->brake * clampf(1.0f + vf / MAX_DRIVE_SPEED, 0.0f, 1.0f);
        }
        if (c->throttle <= 0.01f && c->brake <= 0.01f && !car->boosting) {
            float slow = COAST_DECEL * dt;
            vf = vf > 0.0f ? fmaxf(0.0f, vf - slow) : fminf(0.0f, vf + slow);
        }
        vf = clampf(vf + accel * dt, -MAX_DRIVE_SPEED, MAX_SPEED);
        vl *= expf(-(c->handbrake ? GRIP_HANDBRAKE : GRIP) * dt);

        car->vel = add(mul(fwd, vf), mul(right, vl));
        car->pos = add(car->pos, mul(car->vel, dt));
        car->pos.y = 0.0f;
        car->wheel_distance += vf * dt;
        car->pitch *= expf(-12.0f * dt);
    } else {
        car->pitch = clampf(car->pitch - c->steer_y * AIR_PITCH_RATE * dt, -1.4f, 1.4f);
        car->yaw = wrap_pi(car->yaw - steer * AIR_YAW_RATE * dt);
        if (car->boosting) car->vel = add(car->vel, mul(car_forward(car), BOOST_ACCEL * dt));
        car->vel.y -= GRAVITY * dt;
        float speed = length(car->vel);
        if (speed > MAX_SPEED) car->vel = mul(car->vel, MAX_SPEED / speed);
        car->pos = add(car->pos, mul(car->vel, dt));
        if (car->pos.y <= 0.0f && car->vel.y <= 0.0f) {
            car->pos.y = 0.0f;
            car->vel.y = 0.0f;
            car->on_ground = true;
        }
    }

    arena_push_xz(&car->pos, &car->vel, CAR_RADIUS, 0.25f);
    if (car->pos.y + CAR_TOP > ARENA_HEIGHT) {
        car->pos.y = ARENA_HEIGHT - CAR_TOP;
        if (car->vel.y > 0.0f) car->vel.y = 0.0f;
    }
}

static void ball_step(Ball *ball, float dt)
{
    ball->vel.y -= GRAVITY * dt;
    float drag = expf(-BALL_DRAG * dt);
    ball->vel = mul(ball->vel, drag);
    if (ball->pos.y <= BALL_RADIUS + 0.02f) {
        float roll = expf(-BALL_ROLL_DRAG * dt);
        ball->vel.x *= roll;
        ball->vel.z *= roll;
    }
    ball->pos = add(ball->pos, mul(ball->vel, dt));

    if (ball->pos.y < BALL_RADIUS) {
        ball->pos.y = BALL_RADIUS;
        if (ball->vel.y < 0.0f) ball->vel.y = fabsf(ball->vel.y) < 0.6f ? 0.0f : -ball->vel.y * BALL_BOUNCE;
    }
    if (ball->pos.y > ARENA_HEIGHT - BALL_RADIUS) {
        ball->pos.y = ARENA_HEIGHT - BALL_RADIUS;
        if (ball->vel.y > 0.0f) ball->vel.y = -ball->vel.y * BALL_BOUNCE;
    }
    arena_push_xz(&ball->pos, &ball->vel, BALL_RADIUS, BALL_BOUNCE);

    float speed = length(ball->vel);
    if (speed > BALL_MAX_SPEED) ball->vel = mul(ball->vel, BALL_MAX_SPEED / speed);
}

// A bola contra a caixa do carro, orientada pela frente e pelo nariz.
static void car_hits_ball(Car *car, Ball *ball)
{
    Vec3 fwd = car_forward(car);
    Vec3 right = v3(-cosf(car->yaw), 0.0f, sinf(car->yaw));
    Vec3 up = cross(right, fwd);
    Vec3 offset = sub(ball->pos, add(car->pos, mul(up, CAR_CENTER_HEIGHT)));
    float lx = dot(offset, right), ly = dot(offset, up), lz = dot(offset, fwd);
    float dx = lx - clampf(lx, -CAR_HALF_WIDTH, CAR_HALF_WIDTH);
    float dy = ly - clampf(ly, -CAR_HALF_HEIGHT, CAR_HALF_HEIGHT);
    float dz = lz - clampf(lz, -CAR_HALF_LENGTH, CAR_HALF_LENGTH);
    float dist2 = dx * dx + dy * dy + dz * dz;
    if (dist2 >= BALL_RADIUS * BALL_RADIUS) return;

    Vec3 normal;
    float dist = sqrtf(dist2);
    if (dist > 0.0001f) normal = norm(add(add(mul(right, dx), mul(up, dy)), mul(fwd, dz)));
    else normal = up;                                   // o centro da bola entrou na caixa: empurra para cima
    ball->pos = add(ball->pos, mul(normal, BALL_RADIUS - dist));

    float into = dot(sub(ball->vel, car->vel), normal);
    if (into >= 0.0f) return;
    float share = CAR_MASS / (CAR_MASS + BALL_MASS);
    float change = -(1.0f + HIT_RESTITUTION) * into * share + HIT_EXTRA * length(car->vel);
    ball->vel = add(ball->vel, mul(normal, change));
    // O carro perde um pouco de velocidade no choque.
    car->vel = sub(car->vel, mul(normal, change * (BALL_MASS / CAR_MASS) * 0.5f));
    if (car->on_ground) car->vel.y = 0.0f;
}

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
        cam->yaw = wrap_pi(cam->yaw + wrap_pi(car->yaw - cam->yaw) * blend);
        cam->pitch += (CAMERA_DEFAULT_PITCH - cam->pitch) * blend;
    }

    Vec3 goal = add(car->pos, v3(0.0f, 0.7f, 0.0f));
    cam->target = add(cam->target, mul(sub(goal, cam->target), 1.0f - expf(-18.0f * dt)));
    cam->distance = lerpf(CAMERA_DISTANCE_NEAR, CAMERA_DISTANCE_FAR, clampf(length(car->vel) / MAX_SPEED, 0.0f, 1.0f));

    float cp = cosf(cam->pitch);
    Vec3 back = v3(-sinf(cam->yaw) * cp, sinf(cam->pitch), -cosf(cam->yaw) * cp);
    cam->eye = add(cam->target, mul(back, cam->distance));
    // A camera nao sai do campo nem entra no chao.
    cam->eye.x = clampf(cam->eye.x, -ARENA_HALF_X + 0.5f, ARENA_HALF_X - 0.5f);
    cam->eye.z = clampf(cam->eye.z, -ARENA_HALF_Z + 0.5f, ARENA_HALF_Z - 0.5f);
    cam->eye.y = clampf(cam->eye.y, 0.4f, ARENA_HEIGHT - 0.5f);
}

void game_reset(Game *game)
{
    Car *car = &game->car;
    *car = (Car){0};
    car->pos = v3(-30.0f, 0.0f, 0.0f);
    car->yaw = PI_F * 0.5f;                              // de frente para o gol em +X
    car->boost = 33.0f;
    car->on_ground = true;
    game->ball = (Ball){0};
    game->ball.pos = v3(0.0f, BALL_RADIUS, 0.0f);
}

void game_init(Game *game)
{
    game_reset(game);
    game->camera = (Camera){0};
    game->camera.yaw = game->car.yaw;
    game->camera.pitch = CAMERA_DEFAULT_PITCH;
    game->camera.distance = CAMERA_DISTANCE_NEAR;
    game->camera.target = add(game->car.pos, v3(0.0f, 0.7f, 0.0f));
    camera_update(game, &(Controls){0}, 0.0f);
}

void game_update(Game *game, const Controls *controls, float dt)
{
    dt = clampf(dt, 0.0f, 0.1f);
    if (controls->reset) game_reset(game);
    int steps = (int)ceilf(dt / MAX_STEP);
    if (steps < 1) steps = 1;
    float step = dt / (float)steps;
    for (int i = 0; i < steps; i++) {
        car_step(&game->car, controls, step);
        ball_step(&game->ball, step);
        car_hits_ball(&game->car, &game->ball);
    }
    camera_update(game, controls, dt);
}

Mat4 game_view(const Game *game)
{
    return mat4_look_at(game->camera.eye, game->camera.target);
}
