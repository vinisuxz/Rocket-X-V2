#!/usr/bin/env python3
"""Converte um .glb em um .rxm, o formato binario que o Rocket-X le no Xbox 360.

O que o script faz com o modelo:
  * aplica a transformacao de cada no, entao o modelo fica pronto em coordenadas de mundo (Y para cima);
  * troca texturas por uma cor media e embute a iluminacao na cor de cada vertice (o shader so le posicao e cor);
  * divide os triangulos em pedacos de ate 65535 indices (limite de um draw) e guarda a esfera de cada pedaco,
    para o jogo descartar o que esta fora da tela;
  * separa "partes": com --wheels as rodas do carro (partes 1 a 4, que o jogo gira ao redor do pivo) e com --glass
    os materiais transparentes (parte 1, que o jogo desenha por ultimo, misturado com a imagem).

Formato (big-endian, como o PowerPC do Xbox 360; --little gera little-endian so para testes no PC):
  cabecalho  : "RXM1", u32 partes, u32 pedacos, u32 vertices, u32 indices, f32 min[3], f32 max[3]
  partes     : f32 pivo[3] por parte
  pedacos    : u32 parte, u32 vertice_inicial, u32 vertices, u32 indice_inicial_em_bytes, u32 indices,
               f32 centro[3], f32 raio
  vertices   : f32 x, y, z, r, g, b (24 bytes cada)
  indices    : u16, cada pedaco alinhado em 32 bytes
"""
import argparse
import json
import math
import re
import struct
import sys
from io import BytesIO

import numpy as np

TARGET_TRIANGLES = 12000     # tamanho alvo de um pedaco; pedacos menores deixam o descarte mais preciso
LIGHT_DIR = np.array([0.35, 0.80, 0.48], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)
GLASS_TINT = np.array([0.55, 0.80, 1.00], dtype=np.float32)
WHEEL_ORDER = ["FL", "FR", "BL", "BR"]

COMPONENT_DTYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def load_glb(path):
    data = open(path, "rb").read()
    magic, _version, _length = struct.unpack("<4sII", data[:12])
    if magic != b"glTF":
        sys.exit(f"{path}: nao e um arquivo .glb")
    offset, gltf, blob = 12, None, None
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack("<I4s", data[offset:offset + 8])
        body = data[offset + 8:offset + 8 + chunk_length]
        if chunk_type == b"JSON":
            gltf = json.loads(body)
        elif chunk_type == b"BIN\x00":
            blob = body
        offset += 8 + chunk_length
    return gltf, blob


def read_accessor(gltf, blob, index):
    accessor = gltf["accessors"][index]
    if "sparse" in accessor:
        sys.exit("accessors esparsos nao sao suportados")
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(COMPONENT_DTYPES[accessor["componentType"]])
    width = TYPE_COUNTS[accessor["type"]]
    count = accessor["count"]
    stride = view.get("byteStride") or dtype.itemsize * width
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    array = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=start, strides=(stride, dtype.itemsize))
    return array.astype(np.float32) if dtype == np.float32 else array.astype(np.int64)


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T     # glTF guarda por coluna
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = rotation * s
    m[:3, 3] = t
    return m


def linear_to_display(c):
    return np.power(np.clip(c, 0.0, 1.0), 1.0 / 2.2)


def texture_average(gltf, blob, texture_index, cache):
    """Cor media (ja em sRGB) de uma textura; o jogo nao usa texturas, so essa cor."""
    if texture_index in cache:
        return cache[texture_index]
    from PIL import Image
    image = gltf["images"][gltf["textures"][texture_index]["source"]]
    view = gltf["bufferViews"][image["bufferView"]]
    raw = blob[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
    pixels = np.asarray(Image.open(BytesIO(raw)).convert("RGBA").resize((32, 32)), dtype=np.float32) / 255.0
    weight = pixels[..., 3:4]
    total = weight.sum()
    color = (pixels[..., :3] * weight).sum(axis=(0, 1)) / total if total > 0 else np.array([0.5, 0.5, 0.5], np.float32)
    cache[texture_index] = color.astype(np.float32)
    return cache[texture_index]


TEXTURE_SIZE = 256           # as texturas sao reduzidas para isso antes de amostrar: menos ruido, menos memoria


def texture_pixels(gltf, blob, texture_index, cache):
    """A textura reduzida, como matriz (altura, largura, 3) de 0 a 1, ainda em sRGB."""
    key = ("pixels", texture_index)
    if key not in cache:
        from PIL import Image
        image = gltf["images"][gltf["textures"][texture_index]["source"]]
        view = gltf["bufferViews"][image["bufferView"]]
        raw = blob[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]]
        picture = Image.open(BytesIO(raw)).convert("RGB").resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
        cache[key] = np.asarray(picture, dtype=np.float32) / 255.0
    return cache[key]


def sample_texture(pixels, uv):
    """Amostra bilinear com repeticao; uv no sistema do glTF (origem no canto de cima, sem inverter)."""
    height, width = pixels.shape[:2]
    x = (uv[:, 0] % 1.0) * width - 0.5
    y = (uv[:, 1] % 1.0) * height - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    x0, x1, y0, y1 = x0 % width, (x0 + 1) % width, y0 % height, (y0 + 1) % height
    top = pixels[y0, x0] * (1 - fx) + pixels[y0, x1] * fx
    bottom = pixels[y1, x0] * (1 - fx) + pixels[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def material_colors(gltf, blob, recolor):
    """Por material: (cor base em sRGB, emissao, se e transparente, textura ou None).
    `recolor` troca a cor base pelo nome do material e dispensa a textura."""
    cache, result = {}, []
    for material in gltf.get("materials", []):
        pbr = material.get("pbrMetallicRoughness", {})
        base = linear_to_display(np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=np.float32))
        texture = pbr["baseColorTexture"]["index"] if "baseColorTexture" in pbr else None
        if material.get("name") in recolor:
            base, texture = recolor[material["name"]], None
        pixels = texture_pixels(gltf, blob, texture, cache) if texture is not None else None
        emissive = linear_to_display(np.array(material.get("emissiveFactor", [0, 0, 0]), dtype=np.float32))
        blended = material.get("alphaMode", "OPAQUE") == "BLEND" or pbr.get("baseColorFactor", [1, 1, 1, 1])[3] < 0.9
        result.append((base, emissive, blended, pixels))
    return result


def shade(base, emissive, normals, directional):
    """Iluminacao embutida: hemisferio (igual para qualquer giro do carro) mais uma luz direcional."""
    hemisphere = 0.5 + 0.5 * normals[:, 1:2]
    lambert = np.clip(normals @ LIGHT_DIR, 0.0, 1.0)[:, None]
    light = 0.45 + 0.30 * hemisphere + directional * lambert
    return np.clip(base * light + emissive, 0.0, 1.0)


def collect(gltf, blob, directional, glass, part_of_node, recolor):
    """Percorre a cena e devolve (parte, posicoes, cores, triangulos) em coordenadas de mundo, mais os pivos."""
    materials = material_colors(gltf, blob, recolor)
    default = (np.array([0.6, 0.6, 0.6], np.float32), np.zeros(3, np.float32), False, None)
    pieces, pivots, dropped = [], {}, [0]

    def visit(node_index, parent, part):
        node = gltf["nodes"][node_index]
        world = parent @ node_matrix(node)
        found = part_of_node(node.get("name", ""))
        if found is not None:
            part = found
            pivots[part] = world[:3, 3].copy()
        if "mesh" in node:
            normal_matrix = np.linalg.inv(world[:3, :3]).T
            for primitive in gltf["meshes"][node["mesh"]]["primitives"]:
                if primitive.get("mode", 4) != 4 or "indices" not in primitive:
                    continue
                base, emissive, blended, pixels = materials[primitive["material"]] if "material" in primitive else default
                piece_part = part
                if blended:
                    if not glass:
                        dropped[0] += 1
                        continue
                    piece_part, base, emissive, pixels = 1, GLASS_TINT, np.zeros(3, np.float32), None
                positions = read_accessor(gltf, blob, primitive["attributes"]["POSITION"]).astype(np.float64)
                if "NORMAL" in primitive["attributes"]:
                    normals = read_accessor(gltf, blob, primitive["attributes"]["NORMAL"]).astype(np.float64) @ normal_matrix.T
                    length = np.linalg.norm(normals, axis=1, keepdims=True)
                    normals = (normals / np.where(length > 0, length, 1.0)).astype(np.float32)
                else:
                    normals = np.tile(np.array([[0, 1, 0]], np.float32), (len(positions), 1))
                if pixels is not None and "TEXCOORD_0" in primitive["attributes"]:
                    # a cor da textura vira a cor de cada vertice; o cabecalho do arquivo so guarda posicao e cor
                    base = base * sample_texture(pixels, read_accessor(gltf, blob, primitive["attributes"]["TEXCOORD_0"]))
                world_positions = positions @ world[:3, :3].T + world[:3, 3]
                triangles = read_accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)
                pieces.append((piece_part, world_positions, shade(base, emissive, normals, directional), triangles))
        for child in node.get("children", []):
            visit(child, world, part)

    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        visit(root, np.eye(4), 0)
    return pieces, pivots, dropped[0]


def quad(points, color):
    """Um quadrilatero (4 pontos em ordem) como (parte 0, posicoes, cores, triangulos)."""
    positions = np.array(points, dtype=np.float64)
    return (0, positions, np.tile(np.asarray(color, np.float32), (4, 1)), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64))


def line_quad(x0, z0, x1, z1, width, y, color):
    direction = np.array([x1 - x0, z1 - z0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    side = np.array([-direction[1], direction[0]]) * width / 2
    return quad([(x0 - side[0], y, z0 - side[1]), (x0 + side[0], y, z0 + side[1]),
                 (x1 + side[0], y, z1 + side[1]), (x1 - side[0], y, z1 - side[1])], color)


def pitch_decals(meters, grass, grass_light, line):
    """Faixas do gramado e linhas brancas do campo, como quadrilateros 3 cm acima do piso (y = 0).
    As posicoes saem em metros; o chamador divide pela escala. Valores: o campo plano vai a +-37 m em Z e a +-57,7 m em X."""
    y, half_z, end_x, stripe = 0.03, 36.9, 57.5, 7.5
    pieces = []
    bands = int(90.0 / stripe)
    for i in range(bands):
        if i % 2 == 0:
            continue
        x0 = -45.0 + i * stripe
        pieces.append(quad([(x0, y * 0.5, -half_z), (x0, y * 0.5, half_z), (x0 + stripe, y * 0.5, half_z),
                            (x0 + stripe, y * 0.5, -half_z)], grass_light))
    width = 0.35
    pieces.append(line_quad(0.0, -half_z, 0.0, half_z, width, y, line))
    radius, steps = 9.5, 96
    for i in range(steps):
        a0, a1 = 2 * math.pi * i / steps, 2 * math.pi * (i + 1) / steps
        pieces.append(line_quad(radius * math.cos(a0), radius * math.sin(a0), radius * math.cos(a1), radius * math.sin(a1), width, y, line))
    spot = 0.7
    pieces.append(quad([(-spot, y, -spot), (-spot, y, spot), (spot, y, spot), (spot, y, -spot)], line))
    for sign in (-1.0, 1.0):                              # as areas dos gols
        box_x, box_z = 44.0, 15.0
        pieces.append(line_quad(sign * box_x, -box_z, sign * box_x, box_z, width, y, line))
        for z in (-box_z, box_z):
            pieces.append(line_quad(sign * box_x, z, sign * end_x, z, width, y, line))
    return pieces


def split_triangles(centroids, indices, limit):
    """Divide recursivamente pelo eixo mais longo, no meio, ate cada grupo ter no maximo `limit` triangulos."""
    if len(indices) <= limit:
        return [indices]
    points = centroids[indices]
    axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
    order = indices[np.argsort(points[:, axis], kind="stable")]
    half = len(order) // 2
    return split_triangles(centroids, order[:half], limit) + split_triangles(centroids, order[half:], limit)


def build(pieces, pivots, part_count, scale):
    """Junta tudo, move as partes para o proprio pivo, aplica a escala e divide em pedacos."""
    vertex_blocks, triangle_blocks, part_ids, offset = [], [], [], 0
    for part, positions, colors, triangles in pieces:
        local = (positions - pivots.get(part, 0.0)) * scale
        vertex_blocks.append(np.hstack([local, colors]).astype(np.float32))
        triangle_blocks.append(triangles + offset)
        part_ids.append(np.full(len(triangles), part, dtype=np.int32))
        offset += len(positions)
    vertices = np.vstack(vertex_blocks)
    triangles = np.vstack(triangle_blocks)
    triangle_parts = np.concatenate(part_ids)
    centroids = vertices[triangles][:, :, :3].mean(axis=1)

    chunks = []
    for part in range(part_count):
        selection = np.nonzero(triangle_parts == part)[0]
        if len(selection) == 0:
            continue
        for group in split_triangles(centroids, selection, TARGET_TRIANGLES):
            used, inverse = np.unique(triangles[group].ravel(), return_inverse=True)
            if len(used) > 65535:
                sys.exit("pedaco com vertices demais; reduza TARGET_TRIANGLES")
            chunk_vertices = vertices[used]
            low, high = chunk_vertices[:, :3].min(axis=0), chunk_vertices[:, :3].max(axis=0)
            center = (low + high) / 2
            radius = float(np.linalg.norm(chunk_vertices[:, :3] - center, axis=1).max())
            chunks.append((part, chunk_vertices, inverse.astype(np.uint16), center, radius))
    return chunks


def write_rxm(path, chunks, pivots, part_count, scale, little):
    e = "<" if little else ">"
    all_vertices = np.vstack([c[1] for c in chunks])
    low, high = all_vertices[:, :3].min(axis=0), all_vertices[:, :3].max(axis=0)
    vertex_total = sum(len(c[1]) for c in chunks)
    index_total = sum(len(c[2]) for c in chunks)

    index_blob, table, vertex_start = BytesIO(), [], 0
    for part, chunk_vertices, indices, center, radius in chunks:
        padded = np.zeros((len(indices) + 15) // 16 * 16, dtype=np.uint16)
        padded[:len(indices)] = indices
        table.append(struct.pack(e + "5I4f", part, vertex_start, len(chunk_vertices), index_blob.tell(), len(indices), *center, radius))
        index_blob.write(padded.astype(e + "u2").tobytes())
        vertex_start += len(chunk_vertices)

    out = BytesIO()
    out.write(b"RXM1" + struct.pack(e + "4I6f", part_count, len(chunks), vertex_total, index_total, *low, *high))
    for part in range(part_count):
        out.write(struct.pack(e + "3f", *(pivots.get(part, np.zeros(3)) * scale)))
    out.write(b"".join(table))
    out.write(all_vertices.astype(e + "f4").tobytes())
    out.write(index_blob.getvalue())
    open(path, "wb").write(out.getvalue())
    return low, high, vertex_total, index_total


def rotation_about_y(quarter_turns):
    angle = quarter_turns * math.pi / 2
    c, s = round(math.cos(angle)), round(math.sin(angle))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--scale", type=float, default=1.0, help="escala final, aplicada depois das transformacoes dos nos")
    parser.add_argument("--origin", default=None, help="x,y,z do modelo que vira a origem (antes da escala)")
    parser.add_argument("--directional", type=float, default=0.35, help="forca da luz direcional (0 a 0.5)")
    parser.add_argument("--recolor", action="append", default=[], metavar="MATERIAL=R,G,B",
                        help="troca a cor base de um material (valores de 0 a 1, ja em sRGB); pode repetir")
    parser.add_argument("--glass", action="store_true", help="materiais transparentes viram a parte 1 (por padrao sao descartados)")
    parser.add_argument("--wheels", action="store_true",
                        help="separa as rodas 'Dieci - XX' nas partes 1 a 4 (FL, FR, BL, BR), poe a frente do carro em +Z "
                             "e a origem no chao, entre as rodas")
    parser.add_argument("--pitch", action="store_true", help="desenha faixas e linhas brancas no gramado (so para a arena, em metros antes da escala)")
    parser.add_argument("--little", action="store_true", help="little-endian, so para testes no PC")
    parser.add_argument("--info", action="store_true", help="so mostra as dimensoes, sem escrever arquivo")
    args = parser.parse_args()
    if not args.info and not args.output:
        parser.error("falta o arquivo de saida")

    gltf, blob = load_glb(args.input)
    recolor = {}
    for item in args.recolor:
        name, _, values = item.partition("=")
        recolor[name] = np.array([float(v) for v in values.split(",")], dtype=np.float32)

    def part_of_node(name):
        if args.wheels:
            match = re.match(r"^Dieci - (FL|FR|BL|BR) ", name)
            if match:
                return 1 + WHEEL_ORDER.index(match.group(1))
        return None

    pieces, pivots, dropped = collect(gltf, blob, args.directional, args.glass, part_of_node, recolor)
    part_count = 1
    if args.wheels:
        part_count = 5
    elif args.glass:
        part_count = 2

    rotation = np.eye(3)
    if args.wheels:
        # a frente do carro e de onde estao as rodas dianteiras; gira em Y para ela ficar em +Z
        front = (pivots[1] + pivots[2]) / 2 - (pivots[3] + pivots[4]) / 2
        if abs(front[0]) > abs(front[2]):
            quarter = 3 if front[0] > 0 else 1
        else:
            quarter = 0 if front[2] > 0 else 2
        rotation = rotation_about_y(quarter)
        pieces = [(part, positions @ rotation.T, colors, tri) for part, positions, colors, tri in pieces]
        pivots = {part: rotation @ pivot for part, pivot in pivots.items()}

    stacked = np.vstack([p[1] for p in pieces])
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    if args.origin:
        origin = np.array([float(v) for v in args.origin.split(",")])
    elif args.wheels:
        wheels = np.array([pivots[i] for i in range(1, 5)])
        origin = np.array([wheels[:, 0].mean(), low[1], wheels[:, 2].mean()])
    else:
        origin = np.zeros(3)
    pieces = [(part, positions - origin, colors, tri) for part, positions, colors, tri in pieces]
    pivots = {part: pivot - origin for part, pivot in pivots.items()}

    if args.pitch:
        # as medidas das linhas estao em metros do jogo; aqui as posicoes ainda estao nas unidades do modelo
        pieces += [(part, positions / args.scale, colors, tri) for part, positions, colors, tri in
                   pitch_decals(1.0, np.array([0.09, 0.36, 0.20]), np.array([0.11, 0.41, 0.23]), np.array([0.93, 0.95, 0.97]))]

    print(f"{args.input}: {len(pieces)} primitivas, {dropped} transparentes descartadas")
    print(f"  tamanho {np.round((high - low) * args.scale, 3)}  origem em {np.round(origin, 3)}")
    if args.wheels:
        for i in range(1, 5):
            wheel_pieces = [p[1] - pivots[i] for p in pieces if p[0] == i]
            extent = np.vstack(wheel_pieces)
            print(f"  roda {WHEEL_ORDER[i - 1]}: pivo {np.round(pivots[i] * args.scale, 3)}  "
                  f"raio {round(float((extent[:, 1].max() - extent[:, 1].min()) / 2 * args.scale), 3)}")
    if args.info:
        return

    chunks = build(pieces, pivots, part_count, args.scale)
    low, high, vertex_total, index_total = write_rxm(args.output, chunks, pivots, part_count, args.scale, args.little)
    print(f"{args.output}: {len(chunks)} pedacos, {vertex_total} vertices, {index_total // 3} triangulos, "
          f"limites {np.round(low, 2)} a {np.round(high, 2)}")


if __name__ == "__main__":
    main()
