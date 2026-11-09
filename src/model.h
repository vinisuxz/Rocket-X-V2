#ifndef ROCKETX_MODEL_H
#define ROCKETX_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#include "math3d.h"

// Modelo no formato .rxm, gerado por tools/convert_glb.py (o formato esta descrito la).
// Cada pedaco fica numa alocacao propria da GPU: os vertices (posicao e cor, 24 bytes) e, depois, os indices de 16 bits.

enum {
    MODEL_MAX_PARTS = 5,
    MODEL_VERTEX_BYTES = 24,
};

typedef struct {
    uint32_t part;
    uint32_t vertex_count;
    uint32_t index_count;
    float center[3];
    float radius;
    volatile uint32_t *memory;
    uint32_t index_word_offset;     // onde os indices comecam em `memory`, em palavras de 32 bits
} ModelChunk;

typedef struct {
    uint32_t part_count;
    uint32_t chunk_count;
    Vec3 pivots[MODEL_MAX_PARTS];
    ModelChunk *chunks;
} Model;

typedef struct { float plane[6][4]; } Frustum;

// Le o arquivo para a memoria da GPU. Falso se faltar o arquivo, o formato estiver errado ou acabar a memoria.
bool model_load(Model *model, const char *path);
void model_free(Model *model);

Frustum frustum_from(const Mat4 *view_projection);

// Desenha os pedacos de uma parte que aparecem no frustum. `world` leva a parte ao mundo; `push_register` e
// `push_registers` dizem onde o shader de vertice espera a matriz modelo-vista-projecao. Devolve os pedacos desenhados.
uint32_t model_draw_part(const Model *model, uint32_t part, const Mat4 *view_projection, const Frustum *frustum,
                         const Mat4 *world, uint32_t push_register, uint32_t push_registers);

#endif
