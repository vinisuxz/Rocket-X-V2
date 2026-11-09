#ifndef ROCKETX_GAME_H
#define ROCKETX_GAME_H

#include <stdbool.h>

#include "math3d.h"

// Tudo em metros e segundos. O campo e um retangulo de cantos arredondados no plano XZ (o comprimento corre em X),
// com piso em y = 0 e teto reto. Fica um pouco dentro da cupula de vidro do modelo da arena.
#define ARENA_HALF_X 61.0f
#define ARENA_HALF_Z 40.3f
#define ARENA_HEIGHT 18.0f
#define ARENA_CORNER_RADIUS 10.0f

#define BALL_RADIUS 0.9125f
#define CAR_FRONT_WHEEL_RADIUS 0.133f
#define CAR_REAR_WHEEL_RADIUS 0.153f
#define CAR_MAX_BOOST 100.0f

// O que o jogador pede. Direcao positiva = direita; y positivo = analogico para cima.
typedef struct {
    float steer_x, steer_y;     // analogico esquerdo
    float look_x, look_y;       // analogico direito: orbita a camera
    float throttle, brake;      // gatilhos, de 0 a 1
    bool jump, boost, handbrake, reset;
} Controls;

typedef struct {
    Vec3 pos;                   // ponto de contato com o chao, entre as rodas
    Vec3 vel;
    float yaw;                  // a frente do carro e (sen yaw, 0, cos yaw); yaw cresce para a esquerda
    float pitch;                // so no ar; positivo = nariz para cima
    float boost;                // 0 a CAR_MAX_BOOST
    float wheel_distance;       // metros que as rodas ja rolaram; o jogo divide pelo raio de cada roda
    float steer_visual;         // giro das rodas dianteiras, em radianos, positivo = esquerda
    float jump_time;
    bool on_ground;
    bool boosting;
    bool jump_was_down;
} Car;

typedef struct {
    Vec3 pos;                   // centro da bola
    Vec3 vel;
} Ball;

typedef struct {
    float yaw, pitch, distance; // a camera olha na direcao (sen yaw, 0, cos yaw) e fica `pitch` radianos acima do alvo
    float idle;                 // segundos sem mexer no analogico direito
    Vec3 target, eye;
} Camera;

typedef struct {
    Car car;
    Ball ball;
    Camera camera;
} Game;

void game_init(Game *game);
void game_reset(Game *game);
void game_update(Game *game, const Controls *controls, float dt);
Mat4 game_view(const Game *game);

#endif
