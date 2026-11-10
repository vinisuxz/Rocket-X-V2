#ifndef ROCKETX_GAME_H
#define ROCKETX_GAME_H

#include <stdbool.h>

#include "math3d.h"

// Tudo em metros e segundos. O piso fica em y = 0. O campo e o interior da arena do modelo, medido nele: um retangulo
// de cantos arredondados em XZ (o comprimento corre em X), rampas curvas entre o piso e as paredes, e entre as
// paredes e o teto.
#define ARENA_HALF_X 61.1f
#define ARENA_HALF_Z 40.4f
#define ARENA_HEIGHT 21.6f
#define ARENA_CORNER_RADIUS 16.0f   // dos cantos, vistos de cima
#define ARENA_FILLET_RADIUS 3.4f    // das rampas, vistas de lado

#define BALL_RADIUS 0.9125f
#define CAR_FRONT_WHEEL_RADIUS 0.133f
#define CAR_REAR_WHEEL_RADIUS 0.153f
#define CAR_HULL_CENTER 0.153f      // altura do eixo das rodas (a origem do modelo) acima do chao
#define CAR_MAX_BOOST 100.0f

// A superficie mais proxima dentro da arena: `distance` e positiva para dentro, negativa se o ponto esta na parede, e
// `normal` aponta para dentro do campo.
typedef struct {
    float distance;
    Vec3 normal;
} Surface;

Surface arena_surface(Vec3 point);

// O que o jogador pede. Direcao positiva = direita; y positivo = analogico para cima.
typedef struct {
    float steer_x, steer_y;     // analogico esquerdo
    float look_x, look_y;       // analogico direito: orbita a camera
    float throttle, brake;      // gatilhos, de 0 a 1
    bool jump, boost, handbrake, roll_left, roll_right, reset;
} Controls;

typedef struct {
    Vec3 pos;                   // a origem do modelo: o eixo das rodas
    Vec3 vel;
    Vec3 fwd, up;               // a orientacao; a direita e fwd x up, e a esquerda e o oposto
    Vec3 ang;                   // velocidade angular no ar, no mundo, em rad/s
    Vec3 ground_normal;
    float boost;                // 0 a CAR_MAX_BOOST
    float wheel_distance;       // metros que as rodas ja rolaram; o jogo divide pelo raio de cada roda
    float steer_visual;         // giro das rodas dianteiras, em radianos, positivo = esquerda
    float jump_time;            // quanto falta do empurrao que o pulo segurado ainda da
    float air_lock;             // logo depois de pular, o chao nao conta
    float ground_grace;
    float dodge_window;         // quanto falta para o segundo pulo (mortal) deixar de valer
    float flip_time;            // quanto falta do mortal em andamento
    Vec3 flip_rate;             // velocidade angular do mortal, nos eixos do carro: esquerda, cima, frente
    bool on_ground;
    bool touching;              // algum ponto do casco toca a arena
    bool can_dodge;
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
void game_car_reset(Car *car, Vec3 pos, Vec3 fwd);
void game_update(Game *game, const Controls *controls, float dt);
Mat4 game_view(const Game *game);
// Leva o modelo do carro (origem no chao, entre as rodas; X = esquerda, Y = cima, Z = frente) ao mundo.
Mat4 game_car_matrix(const Car *car);

#endif
