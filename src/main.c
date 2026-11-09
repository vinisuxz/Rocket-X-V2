#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#include "RocketX_shaders.h"
#include "game.h"
#include "math3d.h"
#include "model.h"
#include "xsdk/gpu.h"
#include "xsdk/kernel.h"
#include "xsdk/log.h"
#include "xsdk/present.h"
#include "xsdk/shader.h"

enum {
    CLEAR_ARGB = 0xFF101827,
    BUFFERS = 2,
    SWAP_INTERVAL = 1,
    BEGIN_TIMEOUT_MS = 5000,
    PART_BODY = 0,
    PART_GLASS = 1,                 // na arena, a cupula de vidro
    PART_WHEEL_FL = 1,              // no carro, as rodas: FL, FR, BL, BR
    PART_WHEEL_FR = 2,
    PART_WHEEL_BL = 3,
    PART_WHEEL_BR = 4,
};

static const int32_t DEAD_ZONE = 7000;
static const float FIELD_OF_VIEW = 1.1f;
static const float NEAR_PLANE = 0.4f;
static const float FAR_PLANE = 1500.0f;     // o estadio do modelo vai a mais de 800 m do centro
static const float GLASS_ALPHA = 0.14f;

static XsdkShader vertex_shader, pixel_shader;
static XsdkDepthStencilState depth_state, glass_depth_state;
static XsdkBlendState glass_blend;
static Model arena, car_model, ball_model;
static Game game;

static float stick(int16_t x)
{
    return (x > DEAD_ZONE || x < -DEAD_ZONE) ? (float)x / 32768.0f : 0.0f;
}

static Controls read_controls(const INPUT_STATE *input)
{
    uint16_t buttons = input->gamepad.buttons;
    return (Controls){
        .steer_x = stick(input->gamepad.thumb_lx),
        .steer_y = stick(input->gamepad.thumb_ly),
        .look_x = stick(input->gamepad.thumb_rx),
        .look_y = stick(input->gamepad.thumb_ry),
        .throttle = (float)input->gamepad.right_trigger / 255.0f,
        .brake = (float)input->gamepad.left_trigger / 255.0f,
        .jump = (buttons & INPUT_GAMEPAD_A) != 0,
        .boost = (buttons & INPUT_GAMEPAD_X) != 0,
        .handbrake = (buttons & INPUT_GAMEPAD_B) != 0,
        .reset = (buttons & INPUT_GAMEPAD_Y) != 0,
    };
}

static double seconds_now(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec * 1e-9;
}

static uint32_t draw_part(const Model *model, uint32_t part, const Mat4 *view_projection, const Frustum *frustum,
                          const Mat4 *world)
{
    return model_draw_part(model, part, view_projection, frustum, world, ROCKET_VERT_PUSH, ROCKET_VERT_PUSH_REGISTERS);
}

// Pose do carro: yaw (e nariz, no ar) em torno do ponto de contato com o chao.
static Mat4 car_world(const Car *car)
{
    Mat4 position = mat4_translation(car->pos);
    Mat4 yaw = mat4_rotation_y(car->yaw);
    Mat4 pitch = mat4_rotation_x(-car->pitch);
    Mat4 turned = mat4_multiply(&position, &yaw);
    return mat4_multiply(&turned, &pitch);
}

// Cada roda gira no proprio pivo; as da frente tambem esterçam.
static Mat4 wheel_world(const Model *model, const Mat4 *body, uint32_t part, float steer, float spin)
{
    Mat4 pivot = mat4_translation(model->pivots[part]);
    Mat4 turn = mat4_rotation_y(steer);
    Mat4 roll = mat4_rotation_x(spin);
    Mat4 placed = mat4_multiply(body, &pivot);
    Mat4 steered = mat4_multiply(&placed, &turn);
    return mat4_multiply(&steered, &roll);
}

static void draw_frame(float aspect, uint32_t width, uint32_t height)
{
    Mat4 view = game_view(&game);
    Mat4 projection = mat4_perspective(FIELD_OF_VIEW, aspect, NEAR_PLANE, FAR_PLANE);
    Mat4 view_projection = mat4_multiply(&projection, &view);
    Frustum frustum = frustum_from(&view_projection);

    xsdk_gpu_set_target(width, height);
    xsdk_gpu_set_depth_stencil(&depth_state);
    xsdk_gpu_set_shaders(&vertex_shader, &pixel_shader);
    xsdk_built_shader_set_literals(&rocket_vert);
    xsdk_built_shader_set_literals(&rocket_frag);

    Mat4 identity = mat4_identity();
    draw_part(&arena, PART_BODY, &view_projection, &frustum, &identity);

    const Car *car = &game.car;
    Mat4 body = car_world(car);
    draw_part(&car_model, PART_BODY, &view_projection, &frustum, &body);
    float front_spin = car->wheel_distance / CAR_FRONT_WHEEL_RADIUS;
    float rear_spin = car->wheel_distance / CAR_REAR_WHEEL_RADIUS;
    Mat4 wheel = wheel_world(&car_model, &body, PART_WHEEL_FL, car->steer_visual, front_spin);
    draw_part(&car_model, PART_WHEEL_FL, &view_projection, &frustum, &wheel);
    wheel = wheel_world(&car_model, &body, PART_WHEEL_FR, car->steer_visual, front_spin);
    draw_part(&car_model, PART_WHEEL_FR, &view_projection, &frustum, &wheel);
    wheel = wheel_world(&car_model, &body, PART_WHEEL_BL, 0.0f, rear_spin);
    draw_part(&car_model, PART_WHEEL_BL, &view_projection, &frustum, &wheel);
    wheel = wheel_world(&car_model, &body, PART_WHEEL_BR, 0.0f, rear_spin);
    draw_part(&car_model, PART_WHEEL_BR, &view_projection, &frustum, &wheel);

    Mat4 ball = mat4_translation(game.ball.pos);
    draw_part(&ball_model, PART_BODY, &view_projection, &frustum, &ball);

    // O vidro por ultimo: mistura com o que ja foi desenhado e nao escreve profundidade.
    const float constants[4] = {0.0f, 0.0f, 0.0f, GLASS_ALPHA};
    xsdk_gpu_set_depth_stencil(&glass_depth_state);
    xsdk_gpu_set_blend(&glass_blend);
    xsdk_gpu_set_blend_constants(constants);
    draw_part(&arena, PART_GLASS, &view_projection, &frustum, &identity);
}

static bool load_models(void)
{
    // Os .rxm ficam em assets/models, ao lado do .xex (ver tools/convert_glb.py e o workflow).
    return model_load(&arena, "assets/models/arena.rxm") && model_load(&car_model, "assets/models/car.rxm") &&
           model_load(&ball_model, "assets/models/ball.rxm") && arena.part_count == 2 && car_model.part_count == 5;
}

int main(void)
{
    xsdk_log_write("Rocket-X: starting");
    NTSTATUS status = xsdk_gpu_open();
    if (!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: GPU init failed %08X", (unsigned)status); return 1; }

    status = xsdk_shader_load(&vertex_shader, rocket_vert.container, rocket_vert.container_bytes);
    if (NT_SUCCESS(status)) status = xsdk_shader_load(&pixel_shader, rocket_frag.container, rocket_frag.container_bytes);
    if (!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: shader load failed %08X", (unsigned)status); xsdk_gpu_close(); return 1; }

    const XsdkVertexElement layout[] = {
        {.usage = XSDK_USAGE_POSITION, .format = XSDK_VERTEX_FLOAT3, .offset_bytes = 0},
        {.usage = XSDK_USAGE_COLOR, .format = XSDK_VERTEX_FLOAT3, .offset_bytes = 12}
    };
    status = xsdk_shader_set_vertex_layout(&vertex_shader, layout, 2, (const uint32_t[]){MODEL_VERTEX_BYTES});
    if (!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: vertex layout failed %08X", (unsigned)status); return 1; }

    const XsdkDepthStencil depth = {.depth_test = true, .depth_write = true, .depth_compare = XSDK_COMPARE_LESS_EQUAL};
    const XsdkDepthStencil glass_depth = {.depth_test = true, .depth_write = false, .depth_compare = XSDK_COMPARE_LESS_EQUAL};
    xsdk_gpu_depth_stencil_state(&depth_state, &depth);
    xsdk_gpu_depth_stencil_state(&glass_depth_state, &glass_depth);

    // Mistura com o alfa constante: cor = vidro * alfa + fundo * (1 - alfa).
    const XsdkBlend blend = {
        .color_source = XSDK_BLEND_CONSTANT_ALPHA, .color_dest = XSDK_BLEND_ONE_MINUS_CONSTANT_ALPHA,
        .color_op = XSDK_BLEND_OP_ADD, .alpha_source = XSDK_BLEND_ONE, .alpha_dest = XSDK_BLEND_ZERO,
        .alpha_op = XSDK_BLEND_OP_ADD, .write_mask = XSDK_WRITE_ALL,
    };
    xsdk_gpu_blend_state(&glass_blend, &blend);

    xsdk_log_write("Rocket-X: loading models");
    if (!load_models()) {
        model_free(&ball_model);
        model_free(&car_model);
        model_free(&arena);
        xsdk_gpu_close();
        return 1;
    }

    uint32_t width = 0, height = 0;
    status = xsdk_present_open(XSDK_DISPLAY_720P, BUFFERS, SWAP_INTERVAL, &width, &height);
    if (!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: display init failed %08X", (unsigned)status); return 1; }

    game_init(&game);
    xsdk_log_write("Rocket-X: left stick steer, RT accelerate, LT brake/reverse, A jump, X boost, B handbrake, "
                   "right stick orbit camera, Y reset, Back exit");

    double last = seconds_now();
    for (;;) {
        INPUT_STATE input = {0};
        if (XamInputGetState(0, 0, &input) == ERROR_SUCCESS && (input.gamepad.buttons & INPUT_GAMEPAD_BACK)) break;

        double now = seconds_now();
        Controls controls = read_controls(&input);
        game_update(&game, &controls, (float)(now - last));
        last = now;

        uint32_t address = 0;
        status = xsdk_present_begin(&address, BEGIN_TIMEOUT_MS);
        if (!NT_SUCCESS(status)) { xsdk_log_write("Rocket-X: present begin failed %08X", (unsigned)status); break; }
        draw_frame((float)width / (float)height, width, height);
        xsdk_gpu_resolve(address, width, height, CLEAR_ARGB);
        xsdk_present_end();
    }

    xsdk_present_close();
    model_free(&ball_model);
    model_free(&car_model);
    model_free(&arena);
    xsdk_shader_free(&pixel_shader);
    xsdk_shader_free(&vertex_shader);
    status = xsdk_gpu_close();
    xsdk_log_write("Rocket-X: exit status %08X", (unsigned)status);
    return 0;
}
