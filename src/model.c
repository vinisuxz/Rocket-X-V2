#include "model.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "xsdk/gpu.h"
#include "xsdk/log.h"

enum {
    HEADER_BYTES = 44,          // "RXM1", 4 x u32 e 6 x f32
    CHUNK_RECORD_BYTES = 36,    // 5 x u32 e 4 x f32
    INDEX_BLOCK_ALIGN = 32,
    MAX_CHUNK_VERTICES = 65535,
    MAX_CHUNK_INDICES = 65535,
};

typedef struct {
    uint32_t part, vertex_start, vertex_count, index_offset, index_count;
    float center[3], radius;
} ChunkRecord;

static bool read_exact(int fd, void *buffer, uint32_t bytes)
{
    uint8_t *out = buffer;
    while (bytes) {
        ssize_t got = read(fd, out, bytes);
        if (got <= 0) return false;
        out += got;
        bytes -= (uint32_t)got;
    }
    return true;
}

// A memoria da GPU e write-combined: so se escreve nela com stores de 32 bits.
static void copy_words(volatile uint32_t *destination, const uint32_t *source, uint32_t words)
{
    for (uint32_t i = 0; i < words; i++) destination[i] = source[i];
}

static uint32_t align_up(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

void model_free(Model *model)
{
    if (model->chunks) {
        for (uint32_t i = 0; i < model->chunk_count; i++) xsdk_gpu_free(model->chunks[i].memory);
        free(model->chunks);
    }
    memset(model, 0, sizeof *model);
}

bool model_load(Model *model, const char *path)
{
    memset(model, 0, sizeof *model);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        xsdk_log_write("Rocket-X: cannot open %s", path);
        return false;
    }

    uint32_t header[HEADER_BYTES / 4];
    bool ok = read_exact(fd, header, HEADER_BYTES) && memcmp(header, "RXM1", 4) == 0;
    uint32_t part_count = ok ? header[1] : 0, chunk_count = ok ? header[2] : 0;
    ok = ok && part_count >= 1 && part_count <= MODEL_MAX_PARTS && chunk_count >= 1;

    ChunkRecord *records = NULL;
    uint8_t *vertex_scratch = NULL;
    if (ok) {
        // Depois do cabecalho vem um pivo (3 floats) por parte.
        float pivots[MODEL_MAX_PARTS * 3];
        ok = read_exact(fd, pivots, part_count * 12);
        for (uint32_t i = 0; ok && i < part_count; i++) model->pivots[i] = v3(pivots[i * 3], pivots[i * 3 + 1], pivots[i * 3 + 2]);
        records = malloc((size_t)chunk_count * sizeof *records);
        model->chunks = calloc(chunk_count, sizeof *model->chunks);
        ok = ok && records && model->chunks && read_exact(fd, records, chunk_count * CHUNK_RECORD_BYTES);
    }
    model->part_count = part_count;
    model->chunk_count = chunk_count;

    // Vertices de todos os pedacos, um depois do outro, e so depois os indices.
    uint32_t max_vertices = 0;
    for (uint32_t i = 0; ok && i < chunk_count; i++) {
        const ChunkRecord *r = &records[i];
        ok = r->part < part_count && r->vertex_count >= 3 && r->vertex_count <= MAX_CHUNK_VERTICES &&
             r->index_count >= 3 && r->index_count <= MAX_CHUNK_INDICES;
        if (ok && r->vertex_count > max_vertices) max_vertices = r->vertex_count;
    }
    if (ok) {
        vertex_scratch = malloc((size_t)max_vertices * MODEL_VERTEX_BYTES);
        ok = vertex_scratch != NULL;
    }
    for (uint32_t i = 0; ok && i < chunk_count; i++) {
        const ChunkRecord *r = &records[i];
        ModelChunk *chunk = &model->chunks[i];
        uint32_t vertex_bytes = r->vertex_count * MODEL_VERTEX_BYTES;
        uint32_t index_start = align_up(vertex_bytes, INDEX_BLOCK_ALIGN);
        uint32_t index_bytes = align_up(r->index_count * 2, INDEX_BLOCK_ALIGN);
        chunk->memory = xsdk_gpu_alloc(index_start + index_bytes);
        if (!chunk->memory) {
            xsdk_log_write("Rocket-X: GPU memory ran out loading %s at chunk %u", path, (unsigned)i);
            ok = false;
            break;
        }
        chunk->part = r->part;
        chunk->vertex_count = r->vertex_count;
        chunk->index_count = r->index_count;
        memcpy(chunk->center, r->center, sizeof chunk->center);
        chunk->radius = r->radius;
        chunk->index_word_offset = index_start / 4;
        ok = read_exact(fd, vertex_scratch, vertex_bytes);
        if (ok) copy_words(chunk->memory, (const uint32_t *)vertex_scratch, vertex_bytes / 4);
    }
    free(vertex_scratch);

    // Os indices de cada pedaco ficam no arquivo em blocos alinhados em 32 bytes, na ordem dos pedacos.
    uint32_t *index_scratch = NULL;
    uint32_t max_index_bytes = 0;
    for (uint32_t i = 0; ok && i < chunk_count; i++) {
        uint32_t bytes = align_up(records[i].index_count * 2, INDEX_BLOCK_ALIGN);
        if (bytes > max_index_bytes) max_index_bytes = bytes;
    }
    if (ok) {
        index_scratch = malloc(max_index_bytes);
        ok = index_scratch != NULL;
    }
    for (uint32_t i = 0; ok && i < chunk_count; i++) {
        uint32_t bytes = align_up(records[i].index_count * 2, INDEX_BLOCK_ALIGN);
        ok = read_exact(fd, index_scratch, bytes);
        if (ok) copy_words(model->chunks[i].memory + model->chunks[i].index_word_offset, index_scratch, bytes / 4);
    }
    free(index_scratch);

    for (uint32_t i = 0; ok && i < chunk_count; i++) {
        const ModelChunk *chunk = &model->chunks[i];
        xsdk_gpu_invalidate(xsdk_gpu_address(chunk->memory), (chunk->index_word_offset + align_up(chunk->index_count * 2, INDEX_BLOCK_ALIGN) / 4) * 4);
    }

    free(records);
    close(fd);
    if (!ok) {
        xsdk_log_write("Rocket-X: bad or incomplete model %s", path);
        model_free(model);
    }
    return ok;
}

Frustum frustum_from(const Mat4 *vp)
{
    // Linhas da matriz; a profundidade vai de 0 a 1, entao o plano proximo e a linha 2 sozinha.
    float row[4][4];
    for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) row[r][c] = vp->m[c * 4 + r];
    Frustum f;
    for (int c = 0; c < 4; c++) {
        f.plane[0][c] = row[3][c] + row[0][c];
        f.plane[1][c] = row[3][c] - row[0][c];
        f.plane[2][c] = row[3][c] + row[1][c];
        f.plane[3][c] = row[3][c] - row[1][c];
        f.plane[4][c] = row[2][c];
        f.plane[5][c] = row[3][c] - row[2][c];
    }
    for (int p = 0; p < 6; p++) {
        float n = sqrtf(f.plane[p][0] * f.plane[p][0] + f.plane[p][1] * f.plane[p][1] + f.plane[p][2] * f.plane[p][2]);
        if (n > 0.0f) for (int c = 0; c < 4; c++) f.plane[p][c] /= n;
    }
    return f;
}

static bool sphere_visible(const Frustum *frustum, Vec3 center, float radius)
{
    for (int p = 0; p < 6; p++) {
        const float *plane = frustum->plane[p];
        if (plane[0] * center.x + plane[1] * center.y + plane[2] * center.z + plane[3] < -radius) return false;
    }
    return true;
}

uint32_t model_draw_part(const Model *model, uint32_t part, const Mat4 *view_projection, const Frustum *frustum,
                         const Mat4 *world, uint32_t push_register, uint32_t push_registers)
{
    Mat4 mvp = mat4_multiply(view_projection, world);
    xsdk_gpu_set_vertex_constants(push_register, mvp.m, push_registers);
    uint32_t drawn = 0;
    for (uint32_t i = 0; i < model->chunk_count; i++) {
        const ModelChunk *chunk = &model->chunks[i];
        if (chunk->part != part) continue;
        Vec3 center = mat4_transform_point(world, v3(chunk->center[0], chunk->center[1], chunk->center[2]));
        if (!sphere_visible(frustum, center, chunk->radius)) continue;
        xsdk_gpu_set_vertices(0, xsdk_gpu_address(chunk->memory), chunk->vertex_count * MODEL_VERTEX_BYTES);
        xsdk_gpu_draw_indexed(XSDK_PRIMITIVE_TRIANGLE_LIST, xsdk_gpu_address(chunk->memory + chunk->index_word_offset),
                              chunk->index_count, false);
        drawn++;
    }
    return drawn;
}
