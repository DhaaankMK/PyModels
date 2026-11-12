"""PyModels Studio - editor 3D desktop com CustomTkinter e OpenGL.

Dependencias:
    pip install customtkinter PyOpenGL PyOpenGL_accelerate pyopengltk numpy

Compatibilidade: abre projetos JSON criados pela versao anterior do PyModels.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, colorchooser, messagebox

import numpy as np
import customtkinter as ctk
from OpenGL.GL import *
from OpenGL.GLU import *
from pyopengltk import OpenGLFrame


# ============================================================================
# 0. DIRETORIOS, CONFIGURACAO E CONSTANTES
# ============================================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

PROJECTS_DIR = BASE_DIR / "projects"
EXPORTS_DIR = BASE_DIR / "exports"
CONFIG_PATH = BASE_DIR / "config.json"
AUTOSAVE_PATH = BASE_DIR / "pymodels_autosave.json"
for _folder in (PROJECTS_DIR, EXPORTS_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

APP_VERSION = "2.1.0"
DEFAULT_CONFIG = {
    "window": {"geometry": "1440x850", "appearance": "dark", "theme": "blue"},
    "camera": {"rot_x": 23.0, "rot_y": 38.0, "dist": 10.0, "pan": [0.0, 0.0], "target": [0.0, 0.0, 0.0]},
    "paths": {"last_project": "", "last_export": ""},
    "ui": {
        "grid_visible": True, "gizmo_visible": True, "axes_visible": True,
        "wireframe": False, "snap_enabled": False, "snap_step": 0.5,
        "grid_size": 20, "autosave": True,
    },
}


def deep_merge(base, incoming):
    """Mescla configuracoes mantendo os valores padrao que nao existirem."""
    result = copy.deepcopy(base)
    if not isinstance(incoming, dict):
        return result
    for key, value in incoming.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config():
    if not CONFIG_PATH.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return deep_merge(DEFAULT_CONFIG, json.load(file))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"[config] Nao foi possivel ler config.json: {exc}")
        return copy.deepcopy(DEFAULT_CONFIG)


def save_config(config):
    try:
        temp = CONFIG_PATH.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)
        temp.replace(CONFIG_PATH)
    except OSError as exc:
        print(f"[config] Nao foi possivel salvar configuracao: {exc}")


def clamp(value, low, high):
    return max(low, min(high, value))


def safe_float_list(values, default, minimum=None):
    try:
        result = [float(v) for v in values]
        if len(result) != 3 or not all(math.isfinite(v) for v in result):
            raise ValueError
        if minimum is not None:
            result = [max(minimum, v) for v in result]
        return result
    except (TypeError, ValueError):
        return list(default)


def parse_triplet(text, cast=float):
    # Aceita virgula, ponto e virgula ou espacos entre os valores.
    parts = [p for p in re.split(r"[,;\s]+", text.strip()) if p]
    if len(parts) != 3:
        raise ValueError("Informe exatamente tres valores (X, Y, Z).")
    values = [cast(p) for p in parts]
    return values


# ============================================================================
# 1. OBJETOS E ARVORE DE CENA
# ============================================================================
class SceneNode:
    def __init__(self, name="Objeto", shape="cube", color=(190, 195, 210), parent=None,
                 node_id=None, mesh_vertices=None, mesh_faces=None):
        self.id = node_id or uuid.uuid4().hex[:12]
        self.name = str(name)
        self.shape = shape
        self.color = tuple(int(clamp(int(c), 0, 255)) for c in color[:3])
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.parent = None
        self.children = []
        self.visible = True
        self.locked = False
        self.mesh_vertices = [tuple(map(float, v)) for v in (mesh_vertices or [])]
        self.mesh_faces = [tuple(map(int, f)) for f in (mesh_faces or [])]
        if parent is not None:
            self.set_parent(parent)

    def set_parent(self, parent):
        if self.parent is parent:
            return
        if self.parent and self in self.parent.children:
            self.parent.children.remove(self)
        self.parent = parent
        if parent is not None and self not in parent.children:
            parent.children.append(self)

    def descendants(self):
        yield self
        for child in self.children:
            yield from child.descendants()

    def is_ancestor_of(self, other):
        node = other.parent
        while node is not None:
            if node is self:
                return True
            node = node.parent
        return False


# ============================================================================
# 2. MATEMATICA 3D
# ============================================================================
def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def local_matrix(node):
    rx, ry, rz = [math.radians(a) for a in node.rotation]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rotation = [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]
    matrix = [[0.0] * 4 for _ in range(4)]
    for i in range(3):
        for j in range(3):
            matrix[i][j] = rotation[i][j] * node.scale[j]
    matrix[0][3], matrix[1][3], matrix[2][3] = node.position
    matrix[3][3] = 1.0
    return matrix


def world_matrix(node):
    if node.parent is not None:
        return mat_mul(world_matrix(node.parent), local_matrix(node))
    return local_matrix(node)


def apply_matrix(matrix, vector):
    x, y, z = vector[:3]
    return [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    ]


def mat_inv_affine(matrix):
    linear = [[matrix[i][j] for j in range(3)] for i in range(3)]
    trans = [matrix[i][3] for i in range(3)]
    det = (
        linear[0][0] * (linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1])
        - linear[0][1] * (linear[1][0] * linear[2][2] - linear[1][2] * linear[2][0])
        + linear[0][2] * (linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0])
    )
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    inverse = [
        [(linear[1][1] * linear[2][2] - linear[1][2] * linear[2][1]) * inv_det,
         (linear[0][2] * linear[2][1] - linear[0][1] * linear[2][2]) * inv_det,
         (linear[0][1] * linear[1][2] - linear[0][2] * linear[1][1]) * inv_det],
        [(linear[1][2] * linear[2][0] - linear[1][0] * linear[2][2]) * inv_det,
         (linear[0][0] * linear[2][2] - linear[0][2] * linear[2][0]) * inv_det,
         (linear[0][2] * linear[1][0] - linear[0][0] * linear[1][2]) * inv_det],
        [(linear[1][0] * linear[2][1] - linear[1][1] * linear[2][0]) * inv_det,
         (linear[0][1] * linear[2][0] - linear[0][0] * linear[2][1]) * inv_det,
         (linear[0][0] * linear[1][1] - linear[0][1] * linear[1][0]) * inv_det],
    ]
    offset = [-sum(inverse[i][j] * trans[j] for j in range(3)) for i in range(3)]
    return [inverse[0] + [offset[0]], inverse[1] + [offset[1]], inverse[2] + [offset[2]], [0, 0, 0, 1]]


def decompose_trs(matrix):
    """Decompõe matriz afim em posição, rotação Euler XYZ e escala positiva."""
    columns = [[matrix[row][col] for row in range(3)] for col in range(3)]
    scale = [math.sqrt(sum(value * value for value in column)) for column in columns]
    scale = [max(value, .001) for value in scale]
    rotation = [[matrix[row][col] / scale[col] for col in range(3)] for row in range(3)]
    y = math.asin(clamp(-rotation[2][0], -1.0, 1.0))
    cy = math.cos(y)
    if abs(cy) > 1e-7:
        x = math.atan2(rotation[2][1], rotation[2][2])
        z = math.atan2(rotation[1][0], rotation[0][0])
    else:
        x = math.atan2(-rotation[1][2], rotation[1][1])
        z = 0.0
    position = [matrix[0][3], matrix[1][3], matrix[2][3]]
    angles = [math.degrees(x), math.degrees(y), math.degrees(z)]
    return position, angles, scale


def point_segment_dist(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = clamp(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def triangle_normal(a, b, c):
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


# ============================================================================
# 3. GEOMETRIA / PRIMITIVAS
# ============================================================================
def cube_verts():
    vertices = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, .5, -.5), (-.5, .5, -.5),
                (-.5, -.5, .5), (.5, -.5, .5), (.5, .5, .5), (-.5, .5, .5)]
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
    return vertices, faces


def sphere_verts(rings=16, segments=24):
    vertices, faces = [], []
    for i in range(rings + 1):
        lat = math.pi * i / rings
        for j in range(segments):
            lon = 2 * math.pi * j / segments
            vertices.append((.5 * math.sin(lat) * math.cos(lon), .5 * math.cos(lat),
                             .5 * math.sin(lat) * math.sin(lon)))
    for i in range(rings):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            c, d = (i + 1) * segments + j, (i + 1) * segments + (j + 1) % segments
            faces.extend([(a, b, d), (a, d, c)])
    return vertices, faces


def cylinder_verts(segments=24):
    vertices, faces = [], []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x, z = .5 * math.cos(angle), .5 * math.sin(angle)
        vertices.extend([(x, -.5, z), (x, .5, z)])
    for i in range(segments):
        nxt = (i + 1) % segments
        b0, t0, b1, t1 = i * 2, i * 2 + 1, nxt * 2, nxt * 2 + 1
        faces.extend([(b0, b1, t1), (b0, t1, t0)])
    bottom, top = len(vertices), len(vertices) + 1
    vertices.extend([(0, -.5, 0), (0, .5, 0)])
    for i in range(segments):
        nxt = (i + 1) % segments
        faces.extend([(bottom, nxt * 2, i * 2), (top, i * 2 + 1, nxt * 2 + 1)])
    return vertices, faces


def cone_verts(segments=24):
    vertices, faces = [], []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        vertices.append((.5 * math.cos(angle), -.5, .5 * math.sin(angle)))
    apex = len(vertices)
    vertices.append((0, .5, 0))
    bottom = len(vertices)
    vertices.append((0, -.5, 0))
    for i in range(segments):
        nxt = (i + 1) % segments
        faces.extend([(i, nxt, apex), (bottom, nxt, i)])
    return vertices, faces


def plane_verts():
    return [(-.5, 0, -.5), (.5, 0, -.5), (.5, 0, .5), (-.5, 0, .5)], [(0, 2, 1), (0, 3, 2)]


def pyramid_verts():
    vertices = [(-.5, -.5, -.5), (.5, -.5, -.5), (.5, -.5, .5), (-.5, -.5, .5), (0, .5, 0)]
    faces = [(0, 2, 1), (0, 3, 2), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    return vertices, faces


def torus_verts(major_segments=24, minor_segments=10, major_radius=.34, minor_radius=.16):
    vertices, faces = [], []
    for i in range(major_segments):
        u = 2 * math.pi * i / major_segments
        for j in range(minor_segments):
            v = 2 * math.pi * j / minor_segments
            radius = major_radius + minor_radius * math.cos(v)
            vertices.append((radius * math.cos(u), minor_radius * math.sin(v), radius * math.sin(u)))
    for i in range(major_segments):
        for j in range(minor_segments):
            a = i * minor_segments + j
            b = i * minor_segments + (j + 1) % minor_segments
            c = ((i + 1) % major_segments) * minor_segments + j
            d = ((i + 1) % major_segments) * minor_segments + (j + 1) % minor_segments
            faces.extend([(a, b, d), (a, d, c)])
    return vertices, faces


SHAPES = {
    "cube": cube_verts,
    "sphere": sphere_verts,
    "cylinder": cylinder_verts,
    "cone": cone_verts,
    "plane": plane_verts,
    "pyramid": pyramid_verts,
    "torus": torus_verts,
}
SHAPE_LABELS = {
    "cube": "Cubo", "sphere": "Esfera", "cylinder": "Cilindro", "cone": "Cone",
    "plane": "Plano", "pyramid": "Pirâmide", "torus": "Toro", "empty": "Grupo", "mesh": "Malha importada",
}


def node_geometry(node):
    if node.shape == "mesh":
        return node.mesh_vertices, node.mesh_faces
    if node.shape in SHAPES:
        return SHAPES[node.shape]()
    return [], []


# ============================================================================
# 4. SERIALIZACAO, IMPORTACAO E EXPORTACAO
# ============================================================================
def node_to_dict(node):
    return {
        "id": node.id, "name": node.name, "shape": node.shape, "color": list(node.color),
        "position": list(node.position), "rotation": list(node.rotation), "scale": list(node.scale),
        "visible": node.visible, "locked": node.locked,
        "mesh_vertices": [list(v) for v in node.mesh_vertices],
        "mesh_faces": [list(f) for f in node.mesh_faces],
        "children": [node_to_dict(child) for child in node.children],
    }


def node_from_dict(data, parent=None):
    if not isinstance(data, dict):
        raise ValueError("Estrutura de objeto invalida no projeto.")
    name = str(data.get("name", "Objeto"))
    shape = data.get("shape")
    if shape not in SHAPES and shape not in (None, "empty", "mesh"):
        shape = "cube"
    color = data.get("color", [190, 195, 210])
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        color = [190, 195, 210]
    node = SceneNode(name, shape, color, parent=parent, node_id=data.get("id"),
                     mesh_vertices=data.get("mesh_vertices", []), mesh_faces=data.get("mesh_faces", []))
    node.position = safe_float_list(data.get("position", [0, 0, 0]), [0, 0, 0])
    node.rotation = safe_float_list(data.get("rotation", [0, 0, 0]), [0, 0, 0])
    node.scale = safe_float_list(data.get("scale", [1, 1, 1]), [1, 1, 1], minimum=.001)
    node.visible = bool(data.get("visible", True))
    node.locked = bool(data.get("locked", False))
    for child_data in data.get("children", []):
        node_from_dict(child_data, node)
    return node


def save_project_file(root, path):
    payload = {"format": "PyModels", "version": APP_VERSION, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "scene": node_to_dict(root)}
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    temp.replace(path)


def load_project_file(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    # Formato novo e formato antigo (raiz diretamente no JSON).
    scene_data = data.get("scene") if isinstance(data, dict) and data.get("format") == "PyModels" else data
    if not isinstance(scene_data, dict):
        raise ValueError("O arquivo nao contem uma cena PyModels valida.")
    return node_from_dict(scene_data)


def export_obj(root, obj_path):
    """Exporta a cena inteira em OBJ + MTL, com vertices no espaco global."""
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")
    materials = {}
    vertex_lines, face_lines = [], []
    vertex_offset = 0

    def visit(node):
        nonlocal vertex_offset
        if not node.visible:
            return
        if node.shape in SHAPES or node.shape == "mesh":
            vertices, faces = node_geometry(node)
            matrix = world_matrix(node)
            if vertices and faces:
                material_key = tuple(node.color)
                if material_key not in materials:
                    materials[material_key] = f"material_{len(materials) + 1}"
                safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", node.name).strip("_") or "Objeto"
                face_lines.extend([f"o {safe_name}", f"usemtl {materials[material_key]}"])
                for vertex in vertices:
                    world = apply_matrix(matrix, vertex)
                    vertex_lines.append(f"v {world[0]:.7f} {world[1]:.7f} {world[2]:.7f}")
                for face in faces:
                    if len(face) < 3:
                        continue
                    # Triangula faces maiores que triangulos (fan triangulation).
                    for index in range(1, len(face) - 1):
                        a, b, c = face[0], face[index], face[index + 1]
                        face_lines.append(f"f {a + vertex_offset + 1} {b + vertex_offset + 1} {c + vertex_offset + 1}")
                vertex_offset += len(vertices)
        for child in node.children:
            visit(child)

    visit(root)
    with obj_path.open("w", encoding="utf-8") as file:
        file.write(f"# PyModels Studio {APP_VERSION}\nmtllib {mtl_path.name}\n")
        file.write("\n".join(vertex_lines) + "\n\n")
        file.write("\n".join(face_lines) + "\n")
    with mtl_path.open("w", encoding="utf-8") as file:
        file.write("# Materiais exportados pelo PyModels Studio\n")
        for (r, g, b), name in materials.items():
            file.write(f"newmtl {name}\nKd {r / 255:.5f} {g / 255:.5f} {b / 255:.5f}\nKa 0.08 0.08 0.08\nKs 0.15 0.15 0.15\nNs 24\n\n")
    return obj_path, mtl_path


def import_obj_mesh(path):
    """Importador OBJ basico: v + f, indices positivos/negativos e poligonos."""
    vertices, faces = [], []
    with Path(path).open("r", encoding="utf-8", errors="replace") as file:
        for line_no, raw in enumerate(file, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(tuple(float(value) for value in parts[1:4]))
            elif parts[0] == "f" and len(parts) >= 4:
                face = []
                for token in parts[1:]:
                    index_text = token.split("/")[0]
                    index = int(index_text)
                    index = index - 1 if index > 0 else len(vertices) + index
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"Indice de face invalido na linha {line_no}.")
                    face.append(index)
                for i in range(1, len(face) - 1):
                    faces.append((face[0], face[i], face[i + 1]))
    if not vertices or not faces:
        raise ValueError("O OBJ nao possui vertices e faces compativeis.")
    # Centraliza e normaliza a malha para caber aproximadamente em 1 unidade.
    mins = [min(v[i] for v in vertices) for i in range(3)]
    maxs = [max(v[i] for v in vertices) for i in range(3)]
    center = [(mins[i] + maxs[i]) / 2 for i in range(3)]
    extent = max(maxs[i] - mins[i] for i in range(3)) or 1.0
    vertices = [tuple((v[i] - center[i]) / extent for i in range(3)) for v in vertices]
    return vertices, faces


# ============================================================================
# 5. VIEWPORT OPENGL
# ============================================================================
GIZMO_LEN = 1.35
GIZMO_PICK_PX = 13


class Viewport3D(OpenGLFrame):
    def __init__(self, master, app, **kwargs):
        self.app = app
        self.cam_rot_x, self.cam_rot_y, self.cam_dist = 23.0, 38.0, 10.0
        self.cam_pan = [0.0, 0.0]
        self.cam_target = [0.0, 0.0, 0.0]
        self._orbit_node_id = None
        self._last_mouse = (0, 0)
        self._left_press_pos = None
        self._redraw_job = None
        self._gl_ready = False
        self._last_proj = self._last_mv = self._last_vp = None
        self._drag_axis = None
        self._drag_start_mouse = (0, 0)
        self._drag_start_world = [0.0, 0.0, 0.0]
        self._drag_before = None
        self._drag_start_rotation = [0.0, 0.0, 0.0]
        self._drag_start_scale = [1.0, 1.0, 1.0]
        super().__init__(master, **kwargs)
        self.bind("<ButtonPress-1>", self._on_left_down)
        self.bind("<B1-Motion>", self._on_left_drag)
        self.bind("<ButtonRelease-1>", self._on_left_up)
        self.bind("<ButtonPress-3>", self._on_right_down)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonPress-2>", self._on_middle_down)
        self.bind("<B2-Motion>", self._on_middle_drag)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Button-4>", lambda _e: self._zoom(0.85))
        self.bind("<Button-5>", lambda _e: self._zoom(1.18))
        # O OpenGLFrame redesenha ao expor/redimensionar; os pedidos adicionais
        # passam por tkRedraw() para ativar contexto e trocar os buffers corretamente.
        self.animate = 0
        self.after(180, self._redraw_safe)

    def initgl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 8.0, 6.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.27, 0.27, 0.31, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.88, 0.88, 0.92, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.35, 0.35, 0.4, 1.0])
        glClearColor(0.085, 0.09, 0.12, 1.0)
        glShadeModel(GL_SMOOTH)
        self._gl_ready = True

    def _redraw_safe(self):
        """Agenda um frame OpenGL sem executar render repetido dentro dos eventos.

        O pyopengltk precisa tornar o contexto atual e fazer swap dos buffers.
        Chamar self.redraw() diretamente ignora esse ciclo e pode deixar o
        viewport sem atualizar durante arrastos e alterações de propriedades.
        """
        try:
            if not self.winfo_exists() or self._redraw_job is not None:
                return
            self._redraw_job = self.after(16, self._perform_redraw)
        except tk.TclError:
            self._redraw_job = None

    def _perform_redraw(self):
        self._redraw_job = None
        try:
            if not self.winfo_exists():
                return
            redraw_callback = getattr(self, "tkRedraw", None)
            if callable(redraw_callback):
                redraw_callback()
                return

            # Fallback defensivo para versões/forks do pyopengltk.
            make_current = getattr(self, "tkMakeCurrent", None)
            if callable(make_current):
                make_current()
            if not self._gl_ready:
                self.initgl()
            self.redraw()
            swap_buffers = getattr(self, "tkSwapBuffers", None)
            if callable(swap_buffers):
                swap_buffers()
            else:
                swap_buffers = getattr(self, "tkSwapBuffer", None)
                if callable(swap_buffers):
                    swap_buffers()
        except (tk.TclError, RuntimeError) as exc:
            print(f"[viewport] contexto gráfico indisponível: {exc}")
        except Exception as exc:
            print(f"[viewport] falha ao desenhar: {exc}")

    def redraw(self):
        if not self._gl_ready:
            return
        width, height = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        glViewport(0, 0, width, height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(48.0, width / height, 0.05, 500.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        self._update_orbit_target()
        glTranslatef(self.cam_pan[0], self.cam_pan[1], -self.cam_dist)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        # O foco fica no centro da orbita; a camera gira ao redor dele.
        glTranslatef(-self.cam_target[0], -self.cam_target[1], -self.cam_target[2])
        if self.app.grid_visible:
            self._draw_grid()
        if self.app.axes_visible:
            self._draw_world_axes()
        self._draw_node(self.app.scene_root)
        if self.app.gizmo_visible and self.app.selected_node and not self.app.selected_node.locked:
            self._draw_gizmo(self.app.selected_node)
        self._last_proj = glGetDoublev(GL_PROJECTION_MATRIX)
        self._last_mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        self._last_vp = glGetIntegerv(GL_VIEWPORT)
        glFlush()

    def _update_orbit_target(self):
        """Mantém o foco preso ao objeto enquadrado, mesmo quando ele se move."""
        if not self._orbit_node_id:
            return
        try:
            node = next((item for item in self.app.scene_root.descendants()
                         if item.id == self._orbit_node_id), None)
            if node is None:
                self._orbit_node_id = None
                return
            self.cam_target = apply_matrix(world_matrix(node), [0.0, 0.0, 0.0])
        except (AttributeError, TypeError):
            self._orbit_node_id = None

    def focus_node(self, node, clear_pan=False):
        """Define o objeto como pivô de órbita da câmera."""
        if node is None:
            return
        self._orbit_node_id = node.id
        self.cam_target = apply_matrix(world_matrix(node), [0.0, 0.0, 0.0])
        if clear_pan:
            self.cam_pan = [0.0, 0.0]
        self._redraw_safe()

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        size = int(self.app.grid_size)
        glBegin(GL_LINES)
        for i in range(-size, size + 1):
            if i == 0:
                glColor3f(.35, .48, .68)
            elif i % 5 == 0:
                glColor3f(.22, .25, .32)
            else:
                glColor3f(.15, .17, .22)
            glVertex3f(-size, 0, i)
            glVertex3f(size, 0, i)
            glVertex3f(i, 0, -size)
            glVertex3f(i, 0, size)
        glEnd()
        glEnable(GL_LIGHTING)

    def _draw_world_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, .25, .25); glVertex3f(0, 0, 0); glVertex3f(2.0, 0, 0)
        glColor3f(.25, 1.0, .4); glVertex3f(0, 0, 0); glVertex3f(0, 2.0, 0)
        glColor3f(.3, .55, 1.0); glVertex3f(0, 0, 0); glVertex3f(0, 0, 2.0)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def _draw_node(self, node):
        if not node.visible:
            return
        if node.shape in SHAPES or node.shape == "mesh":
            vertices, faces = node_geometry(node)
            if vertices and faces:
                matrix = world_matrix(node)
                glPushMatrix()
                glMultMatrixf(np.asarray(matrix, dtype=np.float32).flatten(order="F"))
                wire = self.app.wireframe
                if wire:
                    glDisable(GL_LIGHTING)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                    glLineWidth(1.2)
                else:
                    glEnable(GL_LIGHTING)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                if node is self.app.selected_node:
                    glColor3f(1.0, .78, .16)
                else:
                    glColor3f(*(channel / 255.0 for channel in node.color))
                glBegin(GL_TRIANGLES)
                for face in faces:
                    if len(face) < 3:
                        continue
                    a, b, c = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                    glNormal3f(*triangle_normal(a, b, c))
                    glVertex3fv(a); glVertex3fv(b); glVertex3fv(c)
                glEnd()
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                glEnable(GL_LIGHTING)
                glPopMatrix()
        for child in node.children:
            self._draw_node(child)

    def _draw_gizmo(self, node):
        origin = apply_matrix(world_matrix(node), [0, 0, 0])
        axes = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        colors = [(1.0, .22, .22), (.25, 1.0, .35), (.25, .52, 1.0)]
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(2.5)
        if self.app.tool_mode == "rotate":
            radius = 1.22
            for index, (axis, color) in enumerate(zip(axes, colors)):
                glColor3f(*( (1.0, 1.0, .25) if self._drag_axis == index else color ))
                glBegin(GL_LINE_LOOP)
                for step in range(72):
                    angle = 2 * math.pi * step / 72
                    c, sn = math.cos(angle) * radius, math.sin(angle) * radius
                    if index == 0:
                        point = (origin[0], origin[1] + c, origin[2] + sn)
                    elif index == 1:
                        point = (origin[0] + c, origin[1], origin[2] + sn)
                    else:
                        point = (origin[0] + c, origin[1] + sn, origin[2])
                    glVertex3f(*point)
                glEnd()
        else:
            for index, (axis, color) in enumerate(zip(axes, colors)):
                active = self._drag_axis == index
                current = (1.0, 1.0, .25) if active else color
                tip = [origin[i] + axis[i] * GIZMO_LEN for i in range(3)]
                glColor3f(*current)
                glBegin(GL_LINES)
                glVertex3f(*origin)
                glVertex3f(*tip)
                glEnd()
                # Cubo na ponta: seta de movimento ou alca de escala.
                glPushMatrix()
                glTranslatef(*tip)
                radius = .09 if not active else .12
                glBegin(GL_QUADS)
                for face in [
                    [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1)],
                    [(1,-1,1),(-1,-1,1),(-1,1,1),(1,1,1)],
                    [(-1,1,-1),(1,1,-1),(1,1,1),(-1,1,1)],
                    [(-1,-1,1),(1,-1,1),(1,-1,-1),(-1,-1,-1)],
                    [(1,-1,-1),(1,-1,1),(1,1,1),(1,1,-1)],
                    [(-1,-1,1),(-1,-1,-1),(-1,1,-1),(-1,1,1)],
                ]:
                    for vertex in face:
                        glVertex3f(vertex[0] * radius, vertex[1] * radius, vertex[2] * radius)
                glEnd()
                glPopMatrix()
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)

    def project_point(self, point):
        if self._last_proj is None:
            return None
        try:
            x, y, _z = gluProject(point[0], point[1], point[2], self._last_mv, self._last_proj, self._last_vp)
            return x, max(self.winfo_height(), 2) - y
        except Exception:
            return None

    def pick_gizmo(self, sx, sy):
        node = self.app.selected_node
        if not node or not self.app.gizmo_visible or node.locked:
            return None
        origin = apply_matrix(world_matrix(node), [0, 0, 0])
        point0 = self.project_point(origin)
        if point0 is None:
            return None
        if self.app.tool_mode == "rotate":
            radius = 1.22
            for index in range(3):
                projected = []
                for step in range(73):
                    angle = 2 * math.pi * step / 72
                    c, sn = math.cos(angle) * radius, math.sin(angle) * radius
                    if index == 0:
                        point = (origin[0], origin[1] + c, origin[2] + sn)
                    elif index == 1:
                        point = (origin[0] + c, origin[1], origin[2] + sn)
                    else:
                        point = (origin[0] + c, origin[1] + sn, origin[2])
                    screen = self.project_point(point)
                    if screen:
                        projected.append(screen)
                for a, b in zip(projected, projected[1:]):
                    if point_segment_dist(sx, sy, a, b) < 9:
                        return index
            return None
        for index, axis in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
            tip = [origin[i] + axis[i] * GIZMO_LEN for i in range(3)]
            point1 = self.project_point(tip)
            if point1 and point_segment_dist(sx, sy, point0, point1) < GIZMO_PICK_PX:
                return index
        return None

    def pick_object(self, sx, sy):
        best = None
        def visit(node):
            nonlocal best
            if not node.visible:
                return
            if node.shape in SHAPES or node.shape == "mesh":
                matrix = world_matrix(node)
                point = self.project_point(apply_matrix(matrix, [0, 0, 0]))
                if point:
                    distance = math.hypot(point[0] - sx, point[1] - sy)
                    projected_radius = 0.0
                    for axis in ([.5, 0, 0], [0, .5, 0], [0, 0, .5]):
                        edge = self.project_point(apply_matrix(matrix, axis))
                        if edge:
                            projected_radius = max(projected_radius, math.hypot(edge[0] - point[0], edge[1] - point[1]))
                    hit_radius = clamp(projected_radius * 1.2 + 10, 24, 180)
                    if distance < hit_radius and (best is None or distance < best[0]):
                        best = (distance, node)
            for child in node.children:
                visit(child)
        visit(self.app.scene_root)
        return best[1] if best else None

    def _on_left_down(self, event):
        axis = self.pick_gizmo(event.x, event.y)
        if axis is not None:
            self._drag_axis = axis
            self._drag_start_mouse = (event.x, event.y)
            self._drag_before = self.app._scene_snapshot()
            self._drag_start_world = apply_matrix(world_matrix(self.app.selected_node), [0, 0, 0])
            self._drag_start_rotation = list(self.app.selected_node.rotation)
            self._drag_start_scale = list(self.app.selected_node.scale)
            self._redraw_safe()
            return
        self._left_press_pos = (event.x, event.y)
        self._last_mouse = (event.x, event.y)

    def _on_left_drag(self, event):
        if self._drag_axis is not None:
            self._drag_gizmo(event.x, event.y)
            return
        # Arrastar com botao esquerdo orbita a camera.
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self.cam_rot_y += dx * .48
        self.cam_rot_x = clamp(self.cam_rot_x + dy * .48, -89, 89)
        self._last_mouse = (event.x, event.y)
        self._redraw_safe()

    def _on_left_up(self, event):
        if self._drag_axis is not None:
            if self._drag_before is not None:
                self.app._record_snapshot(self._drag_before)
                self.app._mark_dirty()
            self._drag_axis = None
            self._drag_before = None
            self._redraw_safe()
            self.app.refresh_properties()
            return
        press = self._left_press_pos or self._last_mouse
        dx, dy = event.x - press[0], event.y - press[1]
        self._left_press_pos = None
        if abs(dx) < 4 and abs(dy) < 4:
            node = self.pick_object(event.x, event.y)
            self.app.select_node(node, additive=bool(event.state & 0x0001))

    def _drag_gizmo(self, sx, sy):
        node = self.app.selected_node
        if not node or node.locked or self._drag_axis is None:
            return
        axis_index = self._drag_axis
        axis = [(1, 0, 0), (0, 1, 0), (0, 0, 1)][axis_index]
        mdx = sx - self._drag_start_mouse[0]
        mdy = sy - self._drag_start_mouse[1]

        if self.app.tool_mode == "rotate":
            center = self.project_point(self._drag_start_world)
            if center is None:
                return
            start_angle = math.atan2(self._drag_start_mouse[1] - center[1], self._drag_start_mouse[0] - center[0])
            current_angle = math.atan2(sy - center[1], sx - center[0])
            delta = math.degrees(current_angle - start_angle)
            while delta > 180:
                delta -= 360
            while delta < -180:
                delta += 360
            if self.app.snap_enabled:
                step = max(.001, self.app.snap_step)
                delta = round(delta / step) * step
            node.rotation = list(self._drag_start_rotation)
            node.rotation[axis_index] += delta

        elif self.app.tool_mode == "scale":
            point0 = self.project_point(self._drag_start_world)
            point1 = self.project_point([self._drag_start_world[i] + axis[i] * GIZMO_LEN for i in range(3)])
            if not point0 or not point1:
                return
            dx, dy = point1[0] - point0[0], point1[1] - point0[1]
            length = math.hypot(dx, dy)
            if length < 1:
                return
            amount = (mdx * dx / length + mdy * dy / length) / max(length, 1) * 1.8
            if self.app.snap_enabled:
                amount = round(amount / self.app.snap_step) * self.app.snap_step
            node.scale = list(self._drag_start_scale)
            node.scale[axis_index] = max(.001, self._drag_start_scale[axis_index] + amount)

        else:  # move
            point0 = self.project_point(self._drag_start_world)
            point1 = self.project_point([self._drag_start_world[i] + axis[i] * GIZMO_LEN for i in range(3)])
            if not point0 or not point1:
                return
            dx, dy = point1[0] - point0[0], point1[1] - point0[1]
            length = math.hypot(dx, dy)
            if length < 1:
                return
            projected = mdx * (dx / length) + mdy * (dy / length)
            delta = projected * (GIZMO_LEN / length)
            if self.app.snap_enabled:
                step = max(.001, self.app.snap_step)
                delta = round(delta / step) * step
            new_world = [self._drag_start_world[i] + axis[i] * delta for i in range(3)]
            if node.parent:
                inverse = mat_inv_affine(world_matrix(node.parent))
                if inverse is None:
                    return
                node.position = apply_matrix(inverse, new_world)
            else:
                node.position = new_world

        # Atualização visual apenas; o Inspector é atualizado ao soltar o gizmo.
        self._redraw_safe()

    def _on_right_down(self, event):
        self._last_mouse = (event.x, event.y)

    def _on_right_drag(self, event):
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self.cam_rot_y += dx * .48
        self.cam_rot_x = clamp(self.cam_rot_x + dy * .48, -89, 89)
        self._last_mouse = (event.x, event.y)
        self._redraw_safe()

    def _on_middle_down(self, event):
        self._last_mouse = (event.x, event.y)

    def _on_middle_drag(self, event):
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        factor = max(.002, self.cam_dist * .0018)
        self.cam_pan[0] += dx * factor
        self.cam_pan[1] -= dy * factor
        self._last_mouse = (event.x, event.y)
        self._redraw_safe()

    def _on_wheel(self, event):
        self._zoom(.88 if event.delta > 0 else 1.14)

    def _zoom(self, factor):
        self.cam_dist = clamp(self.cam_dist * factor, .5, 300)
        self._redraw_safe()

    def set_camera(self, preset):
        presets = {
            "front": (0, 0), "back": (0, 180), "right": (0, 90), "left": (0, -90),
            "top": (89, 0), "bottom": (-89, 0), "iso": (23, 38),
        }
        self.cam_rot_x, self.cam_rot_y = presets.get(preset, presets["iso"])
        self._redraw_safe()

    def orbit_selected(self):
        if self.app.selected_node is not None:
            self.app.frame_selected()


# ============================================================================
# 6. APLICACAO PRINCIPAL
# ============================================================================
class ModeladorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        ctk.set_appearance_mode(self.cfg["window"].get("appearance", "dark"))
        ctk.set_default_color_theme(self.cfg["window"].get("theme", "blue"))
        self.title(f"PyModels Studio {APP_VERSION} | Editor 3D")
        self.geometry(self._valid_geometry(self.cfg["window"].get("geometry", "1280x780")))
        self.minsize(980, 650)

        self.scene_root = SceneNode("Cena", shape=None)
        self.selected_node = None
        self.selected_nodes = []
        self.tool_mode = "move"  # move, rotate ou scale
        self.current_project = ""
        self.spawn_counter = 1
        self.grid_visible = bool(self.cfg["ui"].get("grid_visible", True))
        self.gizmo_visible = bool(self.cfg["ui"].get("gizmo_visible", True))
        self.axes_visible = bool(self.cfg["ui"].get("axes_visible", True))
        self.wireframe = bool(self.cfg["ui"].get("wireframe", False))
        self.snap_enabled = bool(self.cfg["ui"].get("snap_enabled", False))
        self.snap_step = float(self.cfg["ui"].get("snap_step", .5))
        self.grid_size = int(self.cfg["ui"].get("grid_size", 20))
        self.autosave_enabled = bool(self.cfg["ui"].get("autosave", True))
        self.dirty = False
        self._autosave_job = None
        self._history = []
        self._redo = []
        self._history_limit = 60
        self._building_properties = False
        self._tree_buttons = {}
        self._editor_started = False

        # A primeira tela permite configurar o ambiente antes de abrir a viewport.
        self._build_start_screen()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ START SCREEN
    def _build_start_screen(self):
        self.home_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=("#eef2f7", "#11151d"))
        self.home_frame.pack(fill="both", expand=True)
        self.home_frame.grid_columnconfigure(0, weight=1)
        self.home_frame.grid_rowconfigure(0, weight=1)

        content = ctk.CTkFrame(self.home_frame, fg_color="transparent")
        content.place(relx=.5, rely=.5, anchor="center")

        brand = ctk.CTkFrame(content, fg_color="transparent")
        brand.pack(fill="x", pady=(0, 24))
        ctk.CTkLabel(brand, text="PYMODELS", font=("Segoe UI", 32, "bold"),
                     text_color=("#1d3557", "#e7edf7")).pack()
        ctk.CTkLabel(brand, text=f"STUDIO  •  {APP_VERSION}  •  MODELAGEM 3D",
                     font=("Segoe UI", 11, "bold"), text_color=("#52708f", "#8c9bb0")).pack(pady=(2, 0))
        ctk.CTkLabel(brand, text="O que vamos criar hoje?", font=("Segoe UI", 16),
                     text_color=("#526174", "#b2bdcc")).pack(pady=(13, 0))

        cards = ctk.CTkFrame(content, fg_color="transparent")
        cards.pack()
        cards.grid_columnconfigure(0, weight=1, uniform="home")
        cards.grid_columnconfigure(1, weight=1, uniform="home")

        create_card = ctk.CTkFrame(cards, width=330, height=300, corner_radius=16,
                                   fg_color=("#ffffff", "#1b2330"), border_width=1,
                                   border_color=("#d7e0ec", "#2b3748"))
        create_card.grid(row=0, column=0, padx=9, sticky="nsew")
        create_card.grid_propagate(False)
        ctk.CTkLabel(create_card, text="＋", font=("Segoe UI", 40, "bold"),
                     text_color=("#2878d0", "#6eb4ff")).pack(pady=(25, 0))
        ctk.CTkLabel(create_card, text="Novo objeto", font=("Segoe UI", 20, "bold")).pack(pady=(4, 3))
        ctk.CTkLabel(create_card, text="Comece uma cena do zero e escolha\na primeira forma para modelar.",
                     font=("Segoe UI", 11), justify="center", text_color="gray").pack(pady=(0, 15))
        self.home_shape_var = tk.StringVar(value="Cubo")
        ctk.CTkOptionMenu(create_card, variable=self.home_shape_var,
                          values=["Cubo", "Esfera", "Cilindro", "Cone", "Plano", "Pirâmide", "Toro"],
                          width=220).pack(pady=(0, 12))
        ctk.CTkButton(create_card, text="Criar objeto", width=230, height=42,
                      font=("Segoe UI", 13, "bold"), command=self._create_from_home).pack()

        settings_card = ctk.CTkFrame(cards, width=330, height=300, corner_radius=16,
                                     fg_color=("#ffffff", "#1b2330"), border_width=1,
                                     border_color=("#d7e0ec", "#2b3748"))
        settings_card.grid(row=0, column=1, padx=9, sticky="nsew")
        settings_card.grid_propagate(False)
        ctk.CTkLabel(settings_card, text="⚙", font=("Segoe UI", 35),
                     text_color=("#2878d0", "#6eb4ff")).pack(pady=(22, 0))
        ctk.CTkLabel(settings_card, text="Configurações", font=("Segoe UI", 20, "bold")).pack(pady=(4, 10))

        settings = ctk.CTkFrame(settings_card, fg_color="transparent")
        settings.pack(fill="x", padx=25)
        ctk.CTkLabel(settings, text="Aparência", anchor="w").pack(fill="x", pady=(0, 3))
        appearance_value = self.cfg["window"].get("appearance", "dark").capitalize()
        self.home_appearance_var = tk.StringVar(value=appearance_value if appearance_value in ("Dark", "Light", "System") else "Dark")
        ctk.CTkOptionMenu(settings, variable=self.home_appearance_var,
                          values=["Dark", "Light", "System"], width=250).pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(settings, text="Tema de cores", anchor="w").pack(fill="x", pady=(0, 3))
        theme_map = {"blue": "Azul", "green": "Verde", "dark-blue": "Azul escuro"}
        self.home_theme_var = tk.StringVar(value=theme_map.get(self.cfg["window"].get("theme", "blue"), "Azul"))
        ctk.CTkOptionMenu(settings, variable=self.home_theme_var,
                          values=["Azul", "Verde", "Azul escuro"], width=250).pack(fill="x", pady=(0, 7))
        self.home_autosave_var = tk.BooleanVar(value=self.autosave_enabled)
        ctk.CTkCheckBox(settings, text="Ativar salvamento automático",
                        variable=self.home_autosave_var).pack(anchor="w", pady=(2, 4))
        ctk.CTkButton(settings, text="Aplicar configurações", height=32,
                      command=self._apply_home_settings).pack(fill="x", pady=(3, 0))

        actions = ctk.CTkFrame(content, fg_color="transparent")
        actions.pack(fill="x", pady=(18, 0))
        ctk.CTkButton(actions, text="Abrir projeto", width=190, height=38,
                      fg_color=("#dce8f5", "#26364b"), text_color=("#1d3557", "#e8f1ff"),
                      hover_color=("#cbdced", "#334b68"), command=self._open_project_from_home).pack(side="left", padx=5)
        recent = self.cfg.get("paths", {}).get("last_project", "")
        recent_exists = bool(recent and Path(recent).is_file())
        self.recent_button = ctk.CTkButton(actions, text="Continuar projeto recente", width=220, height=38,
                                           command=self._continue_recent_project,
                                           state="normal" if recent_exists else "disabled")
        self.recent_button.pack(side="left", padx=5)
        ctk.CTkLabel(content, text="Dica: no editor, use G / R / S para mover, girar e escalar.",
                     font=("Segoe UI", 10), text_color="gray").pack(pady=(15, 0))

    def _apply_home_settings(self):
        appearance = self.home_appearance_var.get().lower()
        theme_map = {"Azul": "blue", "Verde": "green", "Azul escuro": "dark-blue"}
        theme = theme_map.get(self.home_theme_var.get(), "blue")
        self.autosave_enabled = bool(self.home_autosave_var.get())
        self.cfg.setdefault("window", {}).update({"appearance": appearance, "theme": theme})
        self.cfg.setdefault("ui", {})["autosave"] = self.autosave_enabled
        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme(theme)
        save_config(self.cfg)
        self.set_status("Configurações salvas.") if self._editor_started else None

    def _create_from_home(self):
        self._apply_home_settings()
        shape_names = {"Cubo": "cube", "Esfera": "sphere", "Cilindro": "cylinder",
                       "Cone": "cone", "Plano": "plane", "Pirâmide": "pyramid", "Toro": "torus"}
        self._launch_editor(mode="new", initial_shape=shape_names.get(self.home_shape_var.get(), "cube"))

    def _open_project_from_home(self):
        path = filedialog.askopenfilename(initialdir=str(PROJECTS_DIR),
            filetypes=[("Projeto PyModels", "*.json"), ("Todos os arquivos", "*.*")],
            title="Abrir projeto")
        if path:
            self._apply_home_settings()
            self._launch_editor(mode="open", project_path=path)

    def _continue_recent_project(self):
        path = self.cfg.get("paths", {}).get("last_project", "")
        if path and Path(path).is_file():
            self._apply_home_settings()
            self._launch_editor(mode="open", project_path=path)

    def _launch_editor(self, mode="new", initial_shape="cube", project_path=None):
        if self._editor_started:
            return
        self._editor_started = True
        try:
            self.home_frame.destroy()
        except tk.TclError:
            pass

        self.scene_root = SceneNode("Cena", shape=None)
        self.selected_node = None
        self.selected_nodes = []
        self.current_project = ""
        self.dirty = False
        self._history.clear(); self._redo.clear()
        self._build_menu()
        self._build_ui()
        self._apply_camera_from_config()

        if mode == "open" and project_path:
            self._startup_scene(load_recent=False, create_demo=False, project_path=project_path)
        elif mode == "recent":
            self._startup_scene(load_recent=True, create_demo=False)
        else:
            self._startup_scene(load_recent=False, create_demo=False)
            self._spawn(initial_shape)
            self.frame_selected()

        self._bind_shortcuts()
        self._schedule_autosave()
        self.after(250, self._update_status)
        self.viewport._redraw_safe()

    @staticmethod
    def _valid_geometry(geometry):
        try:
            if not re.match(r"^\d+x\d+(?:[+-]\d+[+-]\d+)?$", geometry):
                return "1440x850"
            match = re.search(r"([+-]\d+)([+-]\d+)$", geometry)
            if match and (int(match.group(1)) < -100 or int(match.group(2)) < -100):
                return "1440x850"
            return geometry
        except Exception:
            return "1440x850"

    # ------------------------------------------------------------------ MENU
    def _build_menu(self):
        self.menu_bar = tk.Menu(self)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Novo projeto        Ctrl+N", command=self.new_project)
        file_menu.add_command(label="Abrir projeto...     Ctrl+O", command=self._load_project)
        file_menu.add_command(label="Salvar               Ctrl+S", command=self._save_project)
        file_menu.add_command(label="Salvar como...       Ctrl+Shift+S", command=self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command(label="Importar OBJ...", command=self._import_obj)
        file_menu.add_command(label="Exportar OBJ...      Ctrl+E", command=self._export)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self._on_close)
        self.menu_bar.add_cascade(label="Arquivo", menu=file_menu)

        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="Desfazer              Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Refazer                Ctrl+Y", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Duplicar              Ctrl+D", command=self.duplicate_selected)
        edit_menu.add_command(label="Renomear              F2", command=self.rename_selected)
        edit_menu.add_command(label="Excluir               Delete", command=self._delete_selected)
        edit_menu.add_command(label="Agrupar seleção       Ctrl+G", command=self.group_selected)
        edit_menu.add_command(label="Desagrupar            Ctrl+Shift+G", command=self.ungroup_selected)
        edit_menu.add_separator()
        edit_menu.add_command(label="Resetar transformação", command=self.reset_transform)
        self.menu_bar.add_cascade(label="Editar", menu=edit_menu)

        create_menu = tk.Menu(self.menu_bar, tearoff=0)
        for shape, label in [("cube", "Cubo"), ("sphere", "Esfera"), ("cylinder", "Cilindro"),
                             ("cone", "Cone"), ("plane", "Plano"), ("pyramid", "Pirâmide"), ("torus", "Toro")]:
            create_menu.add_command(label=label, command=lambda s=shape: self._spawn(s))
        create_menu.add_separator()
        create_menu.add_command(label="Grupo vazio", command=self.create_empty)
        self.menu_bar.add_cascade(label="Criar", menu=create_menu)

        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="Enquadrar selecionado       F", command=self.frame_selected)
        view_menu.add_command(label="Enquadrar tudo               Home", command=self.frame_all)
        view_menu.add_separator()
        for key, label in [("front", "Frente"), ("back", "Trás"), ("left", "Esquerda"), ("right", "Direita"),
                           ("top", "Superior"), ("bottom", "Inferior"), ("iso", "Isométrica")]:
            view_menu.add_command(label=label, command=lambda p=key: self.viewport.set_camera(p))
        self.menu_bar.add_cascade(label="Visualizar", menu=view_menu)

        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Atalhos do teclado", command=self.show_shortcuts)
        help_menu.add_command(label="Sobre o PyModels Studio", command=self.show_about)
        self.menu_bar.add_cascade(label="Ajuda", menu=help_menu)
        self.configure(menu=self.menu_bar)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.frame_toolbar = ctk.CTkFrame(self, height=48, corner_radius=0)
        self.frame_toolbar.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.frame_toolbar.grid_propagate(False)
        self._toolbar_button("＋ Cubo", lambda: self._spawn("cube"), 78)
        self._toolbar_button("◉ Esfera", lambda: self._spawn("sphere"), 85)
        self._toolbar_button("▤ Cilindro", lambda: self._spawn("cylinder"), 92)
        self._toolbar_button("⌂ Plano", lambda: self._spawn("plane"), 76)
        self._toolbar_button("Mover (G)", lambda: self.set_tool_mode("move"), 88)
        self._toolbar_button("Girar (R)", lambda: self.set_tool_mode("rotate"), 88)
        self._toolbar_button("Escalar (S)", lambda: self.set_tool_mode("scale"), 92)
        self._toolbar_button("Focar objeto (F)", self.frame_selected, 116)
        self._toolbar_button("Duplicar", self.duplicate_selected, 82)
        self._toolbar_button("Excluir", self._delete_selected, 74)
        self._toolbar_button("Desfazer", self.undo, 80)
        self._toolbar_button("Refazer", self.redo, 76)
        self._toolbar_button("Salvar", self._save_project, 72)
        self._toolbar_button("Exportar", self._export, 82)

        self.frame_left = ctk.CTkFrame(self, width=230, corner_radius=0)
        self.frame_left.grid(row=1, column=0, sticky="nsew")
        self.frame_left.grid_propagate(False)
        ctk.CTkLabel(self.frame_left, text="CENA / HIERARQUIA", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 5))
        self.hierarchy_info = ctk.CTkLabel(self.frame_left, text="Shift + clique: seleção múltipla", text_color="gray", font=("Segoe UI", 10))
        self.hierarchy_info.pack(anchor="w", padx=12, pady=(0, 6))
        self.scroll_hier = ctk.CTkScrollableFrame(self.frame_left, width=212, corner_radius=8)
        self.scroll_hier.pack(fill="both", expand=True, padx=7, pady=(0, 8))
        hierarchy_actions = ctk.CTkFrame(self.frame_left, fg_color="transparent")
        hierarchy_actions.pack(fill="x", padx=7, pady=(0, 8))
        ctk.CTkButton(hierarchy_actions, text="+ Grupo", width=95, command=self.create_empty).pack(side="left", padx=2)
        ctk.CTkButton(hierarchy_actions, text="Importar OBJ", width=105, command=self._import_obj).pack(side="left", padx=2)

        self.frame_center = ctk.CTkFrame(self, corner_radius=0, fg_color="#171a21")
        self.frame_center.grid(row=1, column=1, sticky="nsew")
        self.frame_center.grid_rowconfigure(0, weight=1)
        self.frame_center.grid_columnconfigure(0, weight=1)
        self.viewport = Viewport3D(self.frame_center, app=self, width=850, height=650)
        self.viewport.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        overlay = ctk.CTkFrame(self.frame_center, fg_color="#171a21", corner_radius=8)
        overlay.place(relx=.5, rely=.015, anchor="n")
        ctk.CTkLabel(overlay, text="Arraste fora do gizmo para orbitar  •  Botão do meio: mover câmera  •  Scroll: zoom",
                     font=("Segoe UI", 10), text_color="#b6bdca").pack(padx=10, pady=5)
        camera_bar = ctk.CTkFrame(self.frame_center, fg_color="transparent")
        camera_bar.place(relx=.5, rely=.96, anchor="s")
        ctk.CTkButton(camera_bar, text="Orbitar objeto", width=100, height=26,
                      command=self.frame_selected).pack(side="left", padx=3)
        for label, preset in [("Frente", "front"), ("Topo", "top"), ("Lado", "right"), ("Iso", "iso")]:
            ctk.CTkButton(camera_bar, text=label, width=60, height=26, command=lambda p=preset: self.viewport.set_camera(p)).pack(side="left", padx=3)
        self.viewport_hint = ctk.CTkLabel(self.frame_center, text="X vermelho   Y verde   Z azul  |  arraste as alças para mover",
                                          font=("Segoe UI", 10), text_color="#9aa4b2")
        self.viewport_hint.place(relx=.5, rely=.925, anchor="s")

        self.frame_right = ctk.CTkFrame(self, width=305, corner_radius=0)
        self.frame_right.grid(row=1, column=2, sticky="nse")
        self.frame_right.grid_propagate(False)
        ctk.CTkLabel(self.frame_right, text="INSPECTOR", font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 5))
        self.selection_label = ctk.CTkLabel(self.frame_right, text="Nenhum objeto selecionado", text_color="gray")
        self.selection_label.pack(anchor="w", padx=12, pady=(0, 5))
        self.properties_scroll = ctk.CTkScrollableFrame(self.frame_right, width=282, corner_radius=8)
        self.properties_scroll.pack(fill="both", expand=True, padx=7, pady=(0, 8))
        self._build_properties_panel()

        self.frame_bottom = ctk.CTkFrame(self, height=26, corner_radius=0)
        self.frame_bottom.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.frame_bottom.grid_propagate(False)
        self.status_label = ctk.CTkLabel(self.frame_bottom, text="Pronto", anchor="w", font=("Segoe UI", 10))
        self.status_label.pack(side="left", padx=10)
        self.object_count_label = ctk.CTkLabel(self.frame_bottom, text="Objetos: 0", anchor="e", font=("Segoe UI", 10))
        self.object_count_label.pack(side="right", padx=10)

    def _toolbar_button(self, text, command, width):
        ctk.CTkButton(self.frame_toolbar, text=text, width=width, height=31, command=command).pack(side="left", padx=3, pady=8)

    def _section(self, title):
        ctk.CTkLabel(self.properties_scroll, text=title, font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", padx=8, pady=(12, 4))
        ctk.CTkFrame(self.properties_scroll, height=1, fg_color=("#d0d0d0", "#343943")).pack(fill="x", padx=8, pady=(0, 4))

    def _entry_row(self, label, key, placeholder):
        ctk.CTkLabel(self.properties_scroll, text=label, anchor="w", font=("Segoe UI", 10)).pack(fill="x", padx=9, pady=(4, 1))
        entry = ctk.CTkEntry(self.properties_scroll, height=29, placeholder_text=placeholder)
        entry.pack(fill="x", padx=9, pady=(0, 3))
        entry.bind("<Return>", lambda _event, field=key: self._on_property_submit(field))
        return entry

    def _build_properties_panel(self):
        self.entries = {}
        self._section("OBJETO")
        self.entries["name"] = self._entry_row("Nome", "name", "Nome do objeto")
        shape_frame = ctk.CTkFrame(self.properties_scroll, fg_color="transparent")
        shape_frame.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(shape_frame, text="Forma", width=65, anchor="w").pack(side="left")
        self.shape_var = tk.StringVar(value="cube")
        self.shape_menu = ctk.CTkOptionMenu(shape_frame, variable=self.shape_var,
            values=list(SHAPES.keys()), command=self._on_shape_change, width=150)
        self.shape_menu.pack(side="right", fill="x", expand=True)

        self._section("TRANSFORMAÇÃO")
        self.entries["position"] = self._entry_row("Posição X, Y, Z", "position", "0, 0, 0")
        self.entries["rotation"] = self._entry_row("Rotação X, Y, Z", "rotation", "0, 0, 0")
        self.entries["scale"] = self._entry_row("Escala X, Y, Z", "scale", "1, 1, 1")
        transform_buttons = ctk.CTkFrame(self.properties_scroll, fg_color="transparent")
        transform_buttons.pack(fill="x", padx=7, pady=5)
        ctk.CTkButton(transform_buttons, text="Aplicar", width=85, command=self.apply_all_properties).pack(side="left", padx=2)
        ctk.CTkButton(transform_buttons, text="Reset", width=75, command=self.reset_transform).pack(side="left", padx=2)
        ctk.CTkButton(transform_buttons, text="Enquadrar", width=90, command=self.frame_selected).pack(side="left", padx=2)

        self._section("APARÊNCIA")
        color_row = ctk.CTkFrame(self.properties_scroll, fg_color="transparent")
        color_row.pack(fill="x", padx=8, pady=3)
        self.color_preview = ctk.CTkLabel(color_row, text="", width=34, height=28, corner_radius=5, fg_color="#bec3d2")
        self.color_preview.pack(side="left", padx=(0, 7))
        ctk.CTkButton(color_row, text="Escolher cor", command=self.choose_color, height=28).pack(side="left", fill="x", expand=True)
        self.entries["color"] = self._entry_row("RGB (0–255)", "color", "190, 195, 210")
        self.visible_var = tk.BooleanVar(value=True)
        self.locked_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.properties_scroll, text="Objeto visível", variable=self.visible_var,
                        command=self._toggle_selected_visibility).pack(anchor="w", padx=10, pady=(5, 2))
        ctk.CTkCheckBox(self.properties_scroll, text="Bloquear transformação", variable=self.locked_var,
                        command=self._toggle_selected_lock).pack(anchor="w", padx=10, pady=2)

        self._section("FERRAMENTAS")
        tools_grid = ctk.CTkFrame(self.properties_scroll, fg_color="transparent")
        tools_grid.pack(fill="x", padx=6, pady=4)
        actions = [("Duplicar", self.duplicate_selected), ("Renomear", self.rename_selected),
                   ("Agrupar", self.group_selected), ("Desagrupar", self.ungroup_selected),
                   ("Apagar", self._delete_selected), ("Centralizar", self.center_selected)]
        for index, (label, command) in enumerate(actions):
            ctk.CTkButton(tools_grid, text=label, width=125, height=29, command=command).grid(
                row=index // 2, column=index % 2, padx=3, pady=3, sticky="ew")
        tools_grid.grid_columnconfigure(0, weight=1)
        tools_grid.grid_columnconfigure(1, weight=1)

        self._section("VISUALIZAÇÃO")
        self.grid_var = tk.BooleanVar(value=self.grid_visible)
        self.gizmo_var = tk.BooleanVar(value=self.gizmo_visible)
        self.axes_var = tk.BooleanVar(value=self.axes_visible)
        self.wire_var = tk.BooleanVar(value=self.wireframe)
        self.snap_var = tk.BooleanVar(value=self.snap_enabled)
        self.autosave_var = tk.BooleanVar(value=self.autosave_enabled)
        for text, var, callback in [
            ("Mostrar grid", self.grid_var, self._toggle_grid),
            ("Mostrar gizmo", self.gizmo_var, self._toggle_gizmo),
            ("Mostrar eixos XYZ", self.axes_var, self._toggle_axes),
            ("Modo wireframe", self.wire_var, self._toggle_wireframe),
            ("Snap de movimento", self.snap_var, self._toggle_snap),
            ("Salvamento automático", self.autosave_var, self._toggle_autosave),
        ]:
            ctk.CTkCheckBox(self.properties_scroll, text=text, variable=var, command=callback).pack(anchor="w", padx=10, pady=2)
        snap_row = ctk.CTkFrame(self.properties_scroll, fg_color="transparent")
        snap_row.pack(fill="x", padx=9, pady=(4, 10))
        ctk.CTkLabel(snap_row, text="Passo:").pack(side="left")
        self.snap_entry = ctk.CTkEntry(snap_row, width=75, height=27)
        self.snap_entry.insert(0, str(self.snap_step))
        self.snap_entry.pack(side="left", padx=5)
        self.snap_entry.bind("<Return>", self._set_snap_step)
        ctk.CTkButton(snap_row, text="OK", width=42, height=27, command=self._set_snap_step).pack(side="left")

    # ------------------------------------------------------------------ STATE / UNDO
    def _scene_snapshot(self):
        return {"scene": node_to_dict(self.scene_root), "selected": self.selected_node.id if self.selected_node else None}

    def _record_snapshot(self, snapshot):
        self._history.append(copy.deepcopy(snapshot))
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        self._redo.clear()

    def _push_undo(self):
        self._record_snapshot(self._scene_snapshot())

    def _restore_snapshot(self, snapshot):
        self.scene_root = node_from_dict(snapshot["scene"])
        target_id = snapshot.get("selected")
        self.selected_node = next((node for node in self.scene_root.descendants() if node.id == target_id), None)
        self.selected_nodes = [self.selected_node] if self.selected_node else []
        if self.selected_node is not None:
            self.viewport.focus_node(self.selected_node)
        self.refresh_hierarchy()
        self.refresh_properties()
        self.viewport._redraw_safe()
        self._mark_dirty()

    def undo(self):
        if not self._history:
            self.set_status("Nada para desfazer.")
            return
        self._redo.append(self._scene_snapshot())
        self._restore_snapshot(self._history.pop())
        self.set_status("Ação desfeita.")

    def redo(self):
        if not self._redo:
            self.set_status("Nada para refazer.")
            return
        self._history.append(self._scene_snapshot())
        self._restore_snapshot(self._redo.pop())
        self.set_status("Ação refeita.")

    def _mark_dirty(self):
        self.dirty = True
        self.title(f"PyModels Studio {APP_VERSION} | {Path(self.current_project).name if self.current_project else 'Projeto sem título'} •")
        self._schedule_autosave()

    # ------------------------------------------------------------------ HIERARCHY
    def refresh_hierarchy(self):
        for widget in self.scroll_hier.winfo_children():
            widget.destroy()
        self._tree_buttons.clear()

        def add_node(node, depth=0):
            selected = node in self.selected_nodes or node is self.selected_node
            icon = "▣" if node.shape in SHAPES or node.shape == "mesh" else "▾"
            visibility = "◉" if node.visible else "○"
            lock = " 🔒" if node.locked else ""
            text = f"{'  ' * depth}{icon} {node.name}  {visibility}{lock}"
            button = ctk.CTkButton(self.scroll_hier, text=text, anchor="w", height=28,
                fg_color="#315b87" if selected else "transparent",
                hover_color="#293c54", text_color="white",
                command=lambda n=node: self.select_node(n, additive=self._shift_held))
            button.pack(fill="x", pady=1)
            button.bind("<Button-3>", lambda event, n=node: self._hierarchy_context(event, n))
            self._tree_buttons[node.id] = button
            for child in node.children:
                add_node(child, depth + 1)
        for child in self.scene_root.children:
            add_node(child)
        self._update_status()

    def _hierarchy_context(self, event, node):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Selecionar", command=lambda: self.select_node(node))
        menu.add_command(label="Renomear", command=lambda: self.rename_selected(node))
        menu.add_command(label="Duplicar", command=lambda: (self.select_node(node), self.duplicate_selected()))
        menu.add_separator()
        menu.add_command(label="Mostrar / ocultar", command=lambda: self._toggle_node_visibility(node))
        menu.add_command(label="Bloquear / desbloquear", command=lambda: self._toggle_node_lock(node))
        menu.add_separator()
        menu.add_command(label="Excluir", command=lambda: (self.select_node(node), self._delete_selected()))
        menu.tk_popup(event.x_root, event.y_root)

    def select_node(self, node, additive=False):
        if additive and node is not None:
            if node in self.selected_nodes:
                self.selected_nodes.remove(node)
            else:
                self.selected_nodes.append(node)
            self.selected_node = self.selected_nodes[-1] if self.selected_nodes else None
        else:
            self.selected_node = node
            self.selected_nodes = [node] if node else []
        if self.selected_node is not None:
            self.viewport.focus_node(self.selected_node)
        self.refresh_hierarchy()
        self.refresh_properties()
        self.viewport._redraw_safe()

    # ------------------------------------------------------------------ INSPECTOR
    def refresh_properties(self):
        node = self.selected_node
        self._building_properties = True
        for entry in self.entries.values():
            entry.delete(0, "end")
        if not node:
            self.selection_label.configure(text="Nenhum objeto selecionado")
            self.shape_var.set("cube")
            self.visible_var.set(True); self.locked_var.set(False)
            self.color_preview.configure(fg_color="#bec3d2")
            self._building_properties = False
            return
        count = len(self.selected_nodes)
        self.selection_label.configure(text=f"{count} objeto(s) selecionado(s)" if count > 1 else SHAPE_LABELS.get(node.shape, "Objeto"))
        self.entries["name"].insert(0, node.name)
        self.entries["position"].insert(0, ", ".join(f"{v:.3f}" for v in node.position))
        self.entries["rotation"].insert(0, ", ".join(f"{v:.2f}" for v in node.rotation))
        self.entries["scale"].insert(0, ", ".join(f"{v:.3f}" for v in node.scale))
        self.entries["color"].insert(0, ", ".join(str(v) for v in node.color))
        self.shape_var.set(node.shape if node.shape in SHAPES else "cube")
        self.shape_menu.configure(state="normal" if node.shape != "mesh" else "disabled")
        self.visible_var.set(node.visible)
        self.locked_var.set(node.locked)
        self.color_preview.configure(fg_color="#%02x%02x%02x" % node.color)
        self._building_properties = False

    def _on_property_submit(self, key):
        node = self.selected_node
        if not node:
            return
        try:
            if key == "name":
                new_name = self.entries[key].get().strip() or "Objeto"
                if new_name == node.name:
                    return
                self._push_undo(); node.name = new_name
                self.refresh_hierarchy(); self._mark_dirty()
            elif key in ("position", "rotation", "scale"):
                if node.locked:
                    raise ValueError("Este objeto está bloqueado. Desbloqueie-o para alterar a transformação.")
                values = parse_triplet(self.entries[key].get(), float)
                if key == "scale":
                    values = [max(.001, abs(v)) for v in values]
                self._push_undo(); setattr(node, key, values)
                self._mark_dirty()
            elif key == "color":
                values = parse_triplet(self.entries[key].get(), int)
                self._push_undo(); node.color = tuple(int(clamp(v, 0, 255)) for v in values)
                self.color_preview.configure(fg_color="#%02x%02x%02x" % node.color)
                self._mark_dirty()
            self.viewport._redraw_safe()
            self.set_status(f"Propriedade '{key}' atualizada.")
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Valor inválido", str(exc), parent=self)
            self.refresh_properties()

    def apply_all_properties(self):
        if not self.selected_node:
            return
        node = self.selected_node
        try:
            if node.locked:
                raise ValueError("Este objeto está bloqueado. Desbloqueie-o para alterar a transformação.")
            name = self.entries["name"].get().strip() or "Objeto"
            position = parse_triplet(self.entries["position"].get(), float)
            rotation = parse_triplet(self.entries["rotation"].get(), float)
            scale = [max(.001, abs(v)) for v in parse_triplet(self.entries["scale"].get(), float)]
            color = tuple(int(clamp(v, 0, 255)) for v in parse_triplet(self.entries["color"].get(), int))
            self._push_undo()
            node.name, node.position, node.rotation, node.scale, node.color = name, position, rotation, scale, color
            self.refresh_hierarchy(); self.refresh_properties(); self.viewport._redraw_safe(); self._mark_dirty()
            self.set_status("Transformação e aparência aplicadas.")
        except (ValueError, OverflowError) as exc:
            messagebox.showerror("Valores inválidos", str(exc), parent=self)

    def _on_shape_change(self, value):
        if self.selected_node and value in SHAPES and self.selected_node.shape != "mesh":
            self._push_undo(); self.selected_node.shape = value
            self._mark_dirty(); self.viewport._redraw_safe(); self.refresh_hierarchy()

    def choose_color(self):
        if not self.selected_node:
            return
        current = "#%02x%02x%02x" % self.selected_node.color
        chosen = colorchooser.askcolor(color=current, title="Cor do objeto", parent=self)
        if chosen and chosen[0]:
            self._push_undo()
            self.selected_node.color = tuple(int(c) for c in chosen[0])
            self.refresh_properties(); self.viewport._redraw_safe(); self._mark_dirty()

    def _toggle_node_visibility(self, node):
        self._push_undo(); node.visible = not node.visible
        self.refresh_hierarchy(); self.refresh_properties(); self.viewport._redraw_safe(); self._mark_dirty()

    def _toggle_node_lock(self, node):
        self._push_undo(); node.locked = not node.locked
        self.refresh_hierarchy(); self.refresh_properties(); self._mark_dirty()

    def _toggle_selected_visibility(self):
        if self.selected_node and not self._building_properties:
            self._push_undo(); self.selected_node.visible = self.visible_var.get()
            self.refresh_hierarchy(); self.viewport._redraw_safe(); self._mark_dirty()

    def _toggle_selected_lock(self):
        if self.selected_node and not self._building_properties:
            self._push_undo(); self.selected_node.locked = self.locked_var.get()
            self.refresh_hierarchy(); self._mark_dirty()

    # ------------------------------------------------------------------ SCENE ACTIONS
    def _spawn(self, shape):
        if shape not in SHAPES:
            return
        self._push_undo()
        parent = self.selected_node if self.selected_node and self.selected_node.shape in (None, "empty") else self.scene_root
        name = f"{SHAPE_LABELS[shape]}_{self.spawn_counter:03d}"
        self.spawn_counter += 1
        node = SceneNode(name, shape, parent=parent)
        node.position = [0.0, .5, 0.0]
        if self.snap_enabled:
            node.position = [round(v / self.snap_step) * self.snap_step for v in node.position]
        self.select_node(node)
        self._mark_dirty(); self.set_status(f"{SHAPE_LABELS[shape]} criado.")

    def create_empty(self):
        self._push_undo()
        group = SceneNode(f"Grupo_{self.spawn_counter:03d}", "empty", parent=self.scene_root)
        self.spawn_counter += 1
        self.select_node(group); self._mark_dirty()

    def _delete_selected(self):
        nodes = list(self.selected_nodes or ([self.selected_node] if self.selected_node else []))
        nodes = [node for node in nodes if node is not None and node is not self.scene_root]
        # Evita remover um filho duas vezes quando pai e filho estao selecionados.
        roots = [node for node in nodes if not any(other is not node and other.is_ancestor_of(node) for other in nodes)]
        if not roots:
            return
        if not messagebox.askyesno("Excluir objeto(s)", f"Deseja excluir {len(roots)} objeto(s) selecionado(s)?", parent=self):
            return
        self._push_undo()
        for node in roots:
            node.set_parent(None)
        self.select_node(None); self._mark_dirty(); self.set_status("Objeto(s) excluído(s).")

    def duplicate_selected(self):
        if not self.selected_node:
            return
        originals = list(self.selected_nodes or [self.selected_node])
        self._push_undo()
        copies = []
        def clone(source, parent, offset_root=False):
            duplicate = SceneNode(source.name + "_Copy", source.shape, source.color, parent=parent,
                                  mesh_vertices=source.mesh_vertices, mesh_faces=source.mesh_faces)
            duplicate.position = list(source.position); duplicate.rotation = list(source.rotation); duplicate.scale = list(source.scale)
            if offset_root:
                duplicate.position[0] += .75
            duplicate.visible, duplicate.locked = source.visible, source.locked
            for child in source.children:
                clone(child, duplicate, False)
            return duplicate
        for original in originals:
            if original.parent is not None:
                copies.append(clone(original, original.parent, True))
        if copies:
            self.select_node(copies[-1]); self._mark_dirty(); self.set_status(f"{len(copies)} objeto(s) duplicado(s).")

    def rename_selected(self, node=None):
        node = node or self.selected_node
        if not node:
            return
        dialog = ctk.CTkInputDialog(text="Novo nome do objeto:", title="Renomear objeto")
        value = dialog.get_input()
        if value is None:
            return
        value = value.strip()
        if value:
            self._push_undo(); node.name = value
            self.refresh_hierarchy(); self.refresh_properties(); self._mark_dirty()

    def group_selected(self):
        nodes = list(self.selected_nodes or ([self.selected_node] if self.selected_node else []))
        nodes = [node for node in nodes if node.parent is not None]
        if len(nodes) < 2:
            self.set_status("Selecione pelo menos dois objetos para agrupar (Shift + clique).")
            return
        parent = nodes[0].parent
        if any(node.parent is not parent for node in nodes):
            messagebox.showwarning("Agrupar", "Para agrupar, selecione objetos que estejam no mesmo nível da hierarquia.", parent=self)
            return
        self._push_undo()
        group = SceneNode(f"Grupo_{self.spawn_counter:03d}", "empty", parent=parent)
        self.spawn_counter += 1
        for node in nodes:
            node.set_parent(group)  # O grupo nasce na origem, mantendo transformacoes locais.
        self.select_node(group); self._mark_dirty(); self.set_status("Grupo criado.")

    def ungroup_selected(self):
        group = self.selected_node
        if not group or group.shape not in (None, "empty") or not group.children:
            self.set_status("Selecione um grupo com objetos dentro.")
            return
        self._push_undo()
        parent = group.parent or self.scene_root
        parent_inverse = mat_inv_affine(world_matrix(parent))
        if parent_inverse is None:
            self.set_status("Não foi possível desagrupar: transformação do pai inválida.")
            return
        transformed = []
        for child in list(group.children):
            local = mat_mul(parent_inverse, world_matrix(child))
            transformed.append((child, decompose_trs(local)))
        for child, (position, rotation, scale) in transformed:
            child.set_parent(parent)
            child.position, child.rotation, child.scale = position, rotation, scale
        group.set_parent(None)
        self.select_node(None); self._mark_dirty(); self.set_status("Grupo desfeito; transformações preservadas.")

    def reset_transform(self):
        if not self.selected_node:
            return
        self._push_undo()
        node = self.selected_node
        node.position, node.rotation, node.scale = [0, 0, 0], [0, 0, 0], [1, 1, 1]
        self.refresh_properties(); self.viewport._redraw_safe(); self._mark_dirty()

    def center_selected(self):
        if not self.selected_node:
            return
        self._push_undo()
        self.selected_node.position = [0.0, 0.0, 0.0]
        self.refresh_properties(); self.viewport._redraw_safe(); self._mark_dirty()

    # ------------------------------------------------------------------ CAMERA
    def frame_selected(self):
        if not self.selected_node:
            return
        self.viewport.focus_node(self.selected_node, clear_pan=True)
        extent = max((abs(value) for value in self.selected_node.scale), default=1.0)
        self.viewport.cam_dist = clamp(4.2 * max(extent, .5), 2.5, 120)
        self.viewport._redraw_safe()
        self.set_status(f"Câmera focada em: {self.selected_node.name}")

    def frame_all(self):
        nodes = [node for node in self.scene_root.descendants()
                 if node.visible and (node.shape in SHAPES or node.shape == "mesh")]
        if not nodes:
            self.viewport._orbit_node_id = None
            self.viewport.cam_target = [0.0, 0.0, 0.0]
            self.viewport.cam_pan = [0.0, 0.0]
            self.viewport.cam_dist = 10.0
            self.viewport._redraw_safe()
            return
        centers = [apply_matrix(world_matrix(node), [0, 0, 0]) for node in nodes]
        mins = [min(point[i] for point in centers) for i in range(3)]
        maxs = [max(point[i] for point in centers) for i in range(3)]
        center = [(mins[i] + maxs[i]) * .5 for i in range(3)]
        radius = max((math.dist(center, point) for point in centers), default=1.0)
        radius = max(radius, max((max(abs(v) for v in node.scale) * .5 for node in nodes), default=.5))
        self.viewport._orbit_node_id = None
        self.viewport.cam_target = center
        self.viewport.cam_pan = [0.0, 0.0]
        self.viewport.cam_dist = clamp(radius / max(math.sin(math.radians(24)), .1) * 1.18, 4.0, 250)
        self.viewport._redraw_safe()

    # ------------------------------------------------------------------ VIEW OPTIONS
    def set_tool_mode(self, mode):
        if mode not in ("move", "rotate", "scale"):
            return
        self.tool_mode = mode
        labels = {"move": "Mover", "rotate": "Rotacionar", "scale": "Escalar"}
        self.set_status(f"Ferramenta ativa: {labels[mode]}.")
        self.viewport._redraw_safe()

    def _toggle_grid(self):
        self.grid_visible = self.grid_var.get(); self.cfg["ui"]["grid_visible"] = self.grid_visible
        self.viewport._redraw_safe()

    def _toggle_gizmo(self):
        self.gizmo_visible = self.gizmo_var.get(); self.cfg["ui"]["gizmo_visible"] = self.gizmo_visible
        self.viewport._redraw_safe()

    def _toggle_axes(self):
        self.axes_visible = self.axes_var.get(); self.cfg["ui"]["axes_visible"] = self.axes_visible
        self.viewport._redraw_safe()

    def _toggle_wireframe(self):
        self.wireframe = self.wire_var.get(); self.cfg["ui"]["wireframe"] = self.wireframe
        self.viewport._redraw_safe()

    def _toggle_snap(self):
        self.snap_enabled = self.snap_var.get(); self.cfg["ui"]["snap_enabled"] = self.snap_enabled
        self.set_status("Snap ativado." if self.snap_enabled else "Snap desativado.")

    def _toggle_autosave(self):
        self.autosave_enabled = self.autosave_var.get()
        self.cfg["ui"]["autosave"] = self.autosave_enabled
        self._schedule_autosave()
        self.set_status("Salvamento automático ativado." if self.autosave_enabled else "Salvamento automático desativado.")

    def _set_snap_step(self, _event=None):
        try:
            value = float(self.snap_entry.get())
            if not math.isfinite(value) or value <= 0:
                raise ValueError
            self.snap_step = value
            self.cfg["ui"]["snap_step"] = value
            self.set_status(f"Passo do snap: {value:g}")
        except ValueError:
            messagebox.showerror("Snap inválido", "Informe um número maior que zero.", parent=self)
            self.snap_entry.delete(0, "end"); self.snap_entry.insert(0, str(self.snap_step))

    # ------------------------------------------------------------------ PROJECT / IMPORT / EXPORT
    def new_project(self):
        if self.dirty and not messagebox.askyesno("Novo projeto", "Descartar as alterações não salvas?", parent=self):
            return
        self._push_undo()
        self.scene_root = SceneNode("Cena", shape=None)
        self.current_project = ""
        self.select_node(None); self.dirty = False
        self._spawn("cube")
        self.set_status("Novo projeto criado.")

    def _save_project(self):
        if self.current_project:
            return self._save_to_path(self.current_project)
        return self._save_project_as()

    def _save_project_as(self):
        path = filedialog.asksaveasfilename(initialdir=str(PROJECTS_DIR), defaultextension=".json",
            filetypes=[("Projeto PyModels", "*.json"), ("Todos os arquivos", "*.*")], title="Salvar projeto como")
        if path:
            return self._save_to_path(path)
        return False

    def _save_to_path(self, path):
        try:
            save_project_file(self.scene_root, path)
            self.current_project = str(path)
            self.cfg["paths"]["last_project"] = self.current_project
            save_config(self.cfg)
            self.dirty = False
            try:
                AUTOSAVE_PATH.unlink(missing_ok=True)
            except OSError:
                pass
            self.title(f"PyModels Studio {APP_VERSION} | {Path(path).name}")
            self.set_status(f"Projeto salvo: {Path(path).name}")
            return True
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Erro ao salvar", str(exc), parent=self)
            return False

    def _load_project(self):
        path = filedialog.askopenfilename(initialdir=str(PROJECTS_DIR),
            filetypes=[("Projeto PyModels", "*.json"), ("Todos os arquivos", "*.*")], title="Abrir projeto")
        if not path:
            return
        if self.dirty and not messagebox.askyesno("Abrir projeto", "Descartar as alterações não salvas?", parent=self):
            return
        try:
            root = load_project_file(path)
            self.scene_root = root
            self.current_project = str(path)
            self.cfg["paths"]["last_project"] = self.current_project
            save_config(self.cfg)
            self._history.clear(); self._redo.clear(); self.dirty = False
            self.select_node(None); self.frame_all()
            self.set_status(f"Projeto carregado: {Path(path).name}")
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            messagebox.showerror("Erro ao abrir projeto", str(exc), parent=self)

    def _import_obj(self):
        path = filedialog.askopenfilename(initialdir=str(EXPORTS_DIR),
            filetypes=[("Modelo Wavefront OBJ", "*.obj"), ("Todos os arquivos", "*.*")], title="Importar modelo OBJ")
        if not path:
            return
        try:
            vertices, faces = import_obj_mesh(path)
            self._push_undo()
            parent = self.selected_node if self.selected_node and self.selected_node.shape in (None, "empty") else self.scene_root
            node = SceneNode(Path(path).stem, "mesh", (180, 190, 210), parent=parent,
                             mesh_vertices=vertices, mesh_faces=faces)
            node.position = [0, .5, 0]
            self.select_node(node); self._mark_dirty(); self.set_status(f"Modelo importado: {Path(path).name}")
        except (OSError, ValueError, UnicodeError) as exc:
            messagebox.showerror("Falha ao importar OBJ", str(exc), parent=self)

    def _export(self):
        path = filedialog.asksaveasfilename(initialdir=str(EXPORTS_DIR), defaultextension=".obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("Todos os arquivos", "*.*")], title="Exportar cena")
        if not path:
            return
        try:
            obj, mtl = export_obj(self.scene_root, path)
            self.cfg["paths"]["last_export"] = str(obj)
            save_config(self.cfg)
            self.set_status(f"Exportado: {obj.name} + {mtl.name}")
            messagebox.showinfo("Exportação concluída", f"Arquivos criados:\n{obj}\n{mtl}", parent=self)
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Erro ao exportar", str(exc), parent=self)

    # ------------------------------------------------------------------ AUTOSAVE
    def _schedule_autosave(self):
        if self._autosave_job is not None:
            try:
                self.after_cancel(self._autosave_job)
            except tk.TclError:
                pass
        if self.autosave_enabled:
            self._autosave_job = self.after(90000, self._autosave)

    def _autosave(self):
        self._autosave_job = None
        if self.dirty and self.autosave_enabled:
            try:
                save_project_file(self.scene_root, AUTOSAVE_PATH)
                self.set_status("Rascunho de recuperação atualizado.")
            except OSError as exc:
                print(f"[autosave] {exc}")
        self._schedule_autosave()

    def _check_autosave_recovery(self):
        if not AUTOSAVE_PATH.exists():
            return
        try:
            if AUTOSAVE_PATH.stat().st_mtime < time.time() - 7 * 24 * 3600:
                AUTOSAVE_PATH.unlink(missing_ok=True)
                return
            if messagebox.askyesno("Recuperar sessão", "Foi encontrado um rascunho automático. Deseja recuperá-lo?", parent=self):
                self.scene_root = load_project_file(AUTOSAVE_PATH)
                self.current_project = ""
                self.dirty = True
                self.select_node(None); self.frame_all()
                self.set_status("Rascunho automático recuperado. Salve como projeto para manter.")
            else:
                AUTOSAVE_PATH.unlink(missing_ok=True)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[autosave] Nao foi possivel recuperar: {exc}")

    # ------------------------------------------------------------------ SHORTCUTS / STATUS
    def _bind_shortcuts(self):
        self.bind_all("<Control-n>", lambda _e: self.new_project())
        self.bind_all("<Control-o>", lambda _e: self._load_project())
        self.bind_all("<Control-s>", lambda _e: self._save_project())
        self.bind_all("<Control-Shift-S>", lambda _e: self._save_project_as())
        self.bind_all("<Control-e>", lambda _e: self._export())
        self.bind_all("<Control-d>", lambda _e: self.duplicate_selected())
        self.bind_all("<Control-z>", lambda _e: self.undo())
        self.bind_all("<Control-y>", lambda _e: self.redo())
        self.bind_all("<Control-g>", lambda _e: self.group_selected())
        self.bind_all("<Control-Shift-G>", lambda _e: self.ungroup_selected())
        self.bind_all("<Delete>", lambda _e: self._delete_selected())
        self.bind_all("<F2>", lambda _e: self.rename_selected())
        self.bind_all("<f>", lambda _e: self.frame_selected())
        self.bind_all("<KeyPress-g>", lambda event: self._shortcut_tool("move", event))
        self.bind_all("<KeyPress-r>", lambda event: self._shortcut_tool("rotate", event))
        self.bind_all("<KeyPress-s>", lambda event: self._shortcut_tool("scale", event))
        self.bind_all("<Home>", lambda _e: self.frame_all())
        self.bind_all("<Escape>", lambda _e: self.select_node(None))
        self.bind_all("<KeyPress-1>", lambda _e: self._spawn("cube"))
        self.bind_all("<KeyPress-2>", lambda _e: self._spawn("sphere"))
        self.bind_all("<KeyPress-3>", lambda _e: self._spawn("cylinder"))
        self.bind_all("<KeyPress-4>", lambda _e: self._spawn("cone"))
        self.bind_all("<KeyPress-Shift_L>", lambda _e: setattr(self, "_shift_held", True))
        self.bind_all("<KeyPress-Shift_R>", lambda _e: setattr(self, "_shift_held", True))
        self.bind_all("<KeyRelease-Shift_L>", lambda _e: setattr(self, "_shift_held", False))
        self.bind_all("<KeyRelease-Shift_R>", lambda _e: setattr(self, "_shift_held", False))
        self._shift_held = False

    def _shortcut_tool(self, mode, event=None):
        if event is not None and (event.state & 0x0004):
            return
        focused = self.focus_get()
        if isinstance(focused, (tk.Entry, tk.Text)) or (focused is not None and "entry" in focused.winfo_class().lower()):
            return
        self.set_tool_mode(mode)

    def show_shortcuts(self):
        text = (
            "ATALHOS DO PYMODELS STUDIO\n\n"
            "Ctrl + N   Novo projeto\nCtrl + O   Abrir projeto\nCtrl + S   Salvar\nCtrl + Shift + S   Salvar como\n"
            "Ctrl + E   Exportar OBJ\nCtrl + Z / Ctrl + Y   Desfazer / Refazer\nCtrl + D   Duplicar\n"
            "Ctrl + G   Agrupar seleção\nCtrl + Shift + G   Desagrupar\nF2   Renomear\nDelete   Excluir\n"
            "F   Enquadrar selecionado\nHome   Enquadrar cena\nEsc   Limpar seleção\n1–4   Criar primitivas\n"
            "G / R / S   Mover / Rotacionar / Escalar\n\n"
            "Viewport: arraste fora do gizmo para orbitar em torno do foco; botão do meio move; scroll aproxima/afasta.\n"
            "F enquadra a seleção e define o pivô da órbita.\n"
            "Shift + clique na hierarquia ou no viewport adiciona/remove itens da seleção."
        )
        messagebox.showinfo("Atalhos", text, parent=self)

    def show_about(self):
        messagebox.showinfo("Sobre", f"PyModels Studio {APP_VERSION}\nEditor 3D leve para criação de cenas e exportação Wavefront OBJ.", parent=self)

    def set_status(self, message):
        if hasattr(self, "status_label"):
            self.status_label.configure(text=message)

    def _update_status(self):
        if not self.winfo_exists():
            return
        count = sum(1 for node in self.scene_root.descendants() if node.shape in SHAPES or node.shape == "mesh")
        self.object_count_label.configure(text=f"Objetos: {count}  |  Selecionados: {len(self.selected_nodes)}")
        self.after(1000, self._update_status)

    # ------------------------------------------------------------------ STARTUP / SHUTDOWN
    def _apply_camera_from_config(self):
        camera = self.cfg.get("camera", {})
        self.viewport.cam_rot_x = float(camera.get("rot_x", 23.0))
        self.viewport.cam_rot_y = float(camera.get("rot_y", 38.0))
        self.viewport.cam_dist = clamp(float(camera.get("dist", 10.0)), .5, 300)
        pan = camera.get("pan", [0.0, 0.0])
        self.viewport.cam_pan = [float(pan[0]), float(pan[1])] if isinstance(pan, (list, tuple)) and len(pan) == 2 else [0.0, 0.0]
        target = camera.get("target", [0.0, 0.0, 0.0])
        self.viewport.cam_target = safe_float_list(target, [0.0, 0.0, 0.0])

    def _startup_scene(self, load_recent=True, create_demo=True, project_path=None):
        candidate = project_path or (self.cfg["paths"].get("last_project", "") if load_recent else "")
        loaded = False
        if candidate and Path(candidate).is_file():
            try:
                self.scene_root = load_project_file(candidate)
                self.current_project = str(candidate)
                self.cfg.setdefault("paths", {})["last_project"] = str(candidate)
                loaded = True
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"[startup] Falha ao abrir projeto: {exc}")
                if project_path:
                    messagebox.showerror("Erro ao abrir projeto", str(exc), parent=self)
        if not loaded and create_demo:
            demo = SceneNode("Cubo_Demo", "cube", (90, 145, 220), parent=self.scene_root)
            demo.position = [0, .5, 0]
            self.selected_node = demo
            self.selected_nodes = [demo]
        elif loaded:
            visible = [node for node in self.scene_root.descendants() if node.shape in SHAPES or node.shape == "mesh"]
            if visible:
                self.selected_node = visible[0]
                self.selected_nodes = [visible[0]]
        self.refresh_hierarchy(); self.refresh_properties()
        if loaded and self.selected_node:
            # Mantém a orientação/distance salvas, mas usa a seleção como pivô de órbita.
            self.viewport.focus_node(self.selected_node)
        elif create_demo and self.selected_node:
            self.frame_selected()
        self.after(350, self._check_autosave_recovery)

    def _on_close(self):
        if not self._editor_started:
            self.cfg.setdefault("window", {})["appearance"] = self.home_appearance_var.get().lower()
            theme_map = {"Azul": "blue", "Verde": "green", "Azul escuro": "dark-blue"}
            self.cfg["window"]["theme"] = theme_map.get(self.home_theme_var.get(), "blue")
            self.cfg.setdefault("ui", {})["autosave"] = bool(self.home_autosave_var.get())
            save_config(self.cfg)
            self.destroy()
            return
        if self.dirty:
            choice = messagebox.askyesnocancel("Sair do PyModels", "Existem alterações não salvas. Deseja salvar antes de sair?", parent=self)
            if choice is None:
                return
            if choice and not self._save_project():
                return
            if not choice and self.autosave_enabled:
                try:
                    save_project_file(self.scene_root, AUTOSAVE_PATH)
                except OSError as exc:
                    print(f"[autosave] Nao foi possivel gravar rascunho ao sair: {exc}")
        self.cfg["window"]["geometry"] = self.geometry()
        self.cfg["window"]["appearance"] = ctk.get_appearance_mode().lower()
        self.cfg["camera"].update({
            "rot_x": self.viewport.cam_rot_x, "rot_y": self.viewport.cam_rot_y,
            "dist": self.viewport.cam_dist, "pan": list(self.viewport.cam_pan),
            "target": list(self.viewport.cam_target),
        })
        self.cfg["ui"].update({
            "grid_visible": self.grid_visible, "gizmo_visible": self.gizmo_visible,
            "axes_visible": self.axes_visible, "wireframe": self.wireframe,
            "snap_enabled": self.snap_enabled, "snap_step": self.snap_step,
            "grid_size": self.grid_size, "autosave": self.autosave_enabled,
        })
        save_config(self.cfg)
        if self._autosave_job is not None:
            try:
                self.after_cancel(self._autosave_job)
            except tk.TclError:
                pass
        self.destroy()


if __name__ == "__main__":
    app = ModeladorApp()
    app.mainloop()
