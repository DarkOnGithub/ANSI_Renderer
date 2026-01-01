import math
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config, DEVICE, HIDE_CURSOR, SHOW_CURSOR, CLEAR_SCREEN, ENABLE_ALT_BUFFER, DISABLE_ALT_BUFFER
from src.ansi_renderer import AnsiRenderer

FPS = 45

vs = [
    {"x": -0.25, "y": -0.25, "z": -0.25},
    {"x":  0.25, "y": -0.25, "z": -0.25},
    {"x":  0.25, "y":  0.25, "z": -0.25},
    {"x": -0.25, "y":  0.25, "z": -0.25},
    {"x": -0.25, "y": -0.25, "z":  0.25},
    {"x":  0.25, "y": -0.25, "z":  0.25},
    {"x":  0.25, "y":  0.25, "z":  0.25},
    {"x": -0.25, "y":  0.25, "z":  0.25},
]

fs = [
    [4, 5, 6, 7], 
    [1, 0, 3, 2], 
    [7, 6, 2, 3], 
    [0, 1, 5, 4], 
    [0, 4, 7, 3], 
    [5, 1, 2, 6], 
]

def rotate_yz(p, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return {
        "x": p["x"],
        "y": p["y"] * c - p["z"] * s,
        "z": p["y"] * s + p["z"] * c,
    }

def rotate_xz(p, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return {
        "x": p["x"] * c - p["z"] * s,
        "y": p["y"],
        "z": p["x"] * s + p["z"] * c,
    }

def rotate_xy(p, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return {
        "x": p["x"] * c - p["y"] * s,
        "y": p["x"] * s + p["y"] * c,
        "z": p["z"],
    }

def translate_z(p, dz):
    return {"x": p["x"], "y": p["y"], "z": p["z"] + dz}

def project(p):
    z = p["z"]
    if z == 0: z = 0.001
    return {
        "x": p["x"] / z,
        "y": p["y"] / z,
        "z": z 
    }

def screen(p, width, height):
    return {
        "x": int((p["x"] + 1) / 2 * (width - 1)),
        "y": int((1 - (p["y"] + 1) / 2) * (height - 1)),
        "z": p["z"] 
    }

def fill_triangle(buffer, z_buffer, p0, p1, p2, c0, c1, c2, width, height):
    min_x = max(0, int(min(p0["x"], p1["x"], p2["x"])))
    max_x = min(width - 1, int(max(p0["x"], p1["x"], p2["x"])))
    min_y = max(0, int(min(p0["y"], p1["y"], p2["y"])))
    max_y = min(height - 1, int(max(p0["y"], p1["y"], p2["y"])))

    if min_x > max_x or min_y > max_y:
        return

    yy, xx = torch.meshgrid(
        torch.arange(min_y, max_y + 1, device=DEVICE),
        torch.arange(min_x, max_x + 1, device=DEVICE),
        indexing='ij'
    )

    v0x, v0y = p1["x"] - p0["x"], p1["y"] - p0["y"]
    v1x, v1y = p2["x"] - p0["x"], p2["y"] - p0["y"]
    v2x, v2y = xx - p0["x"], yy - p0["y"]

    den = v0x * v1y - v1x * v0y
    if abs(den) < 1e-6:
        return

    v = (v2x * v1y - v1x * v2y) / den
    w = (v0x * v2y - v2x * v0y) / den
    u = 1.0 - v - w

    mask = (u >= 0) & (v >= 0) & (w >= 0)
    
    z_interp = u * p0["z"] + v * p1["z"] + w * p2["z"]
    
    current_z = z_buffer[min_y:max_y+1, min_x:max_x+1]
    depth_mask = mask & (z_interp < current_z)
    
    current_z[depth_mask] = z_interp[depth_mask]
    
    u_m = u[depth_mask].unsqueeze(-1)
    v_m = v[depth_mask].unsqueeze(-1)
    w_m = w[depth_mask].unsqueeze(-1)
    
    colors_interp = (u_m * c0 + v_m * c1 + w_m * c2).to(torch.uint8)
    buffer[min_y:max_y+1, min_x:max_x+1][depth_mask] = colors_interp

def draw_line(buffer, p0, p1, c0, c1, width, height):
    x0, y0 = int(p0["x"]), int(p0["y"])
    x1, y1 = int(p1["x"]), int(p1["y"])
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    color = c0.to(torch.uint8) 
    
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            buffer[y0, x0] = color
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def get_terminal_size():
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except OSError:
        return 80, 40

def render_frame_to_tensor(angle, dz, width, height):
    buffer = torch.zeros((height, width, 3), dtype=torch.uint8, device=DEVICE)
    z_buffer = torch.full((height, width), float("inf"), device=DEVICE)
    
    t_vals = torch.arange(len(vs), device=DEVICE, dtype=torch.float32)
    r = 127.5 + 127.5 * torch.sin(angle + t_vals * 0.5)
    g = 127.5 + 127.5 * torch.sin(angle * 1.5 + t_vals * 0.7 + 2.0)
    b = 127.5 + 127.5 * torch.sin(angle * 0.7 + t_vals * 0.9 + 4.0)
    dynamic_vertex_colors = torch.stack([r, g, b], dim=1)
    
    for f_idx, f in enumerate(fs):
        if len(f) < 3:
            continue 
            
        face_vertices = []
        face_vertex_colors = []
        for idx in f:
            v_raw = vs[idx]
            v = rotate_yz(v_raw, angle)
            v = rotate_xz(v, angle * 0.7)
            v = rotate_xy(v, angle * 0.3)
            v = translate_z(v, dz)
            face_vertices.append(screen(project(v), width, height))
            face_vertex_colors.append(dynamic_vertex_colors[idx])

        for i in range(1, len(face_vertices) - 1):
            fill_triangle(
                buffer, z_buffer, 
                face_vertices[0], 
                face_vertices[i], 
                face_vertices[i+1], 
                face_vertex_colors[0],
                face_vertex_colors[i],
                face_vertex_colors[i+1],
                width, height
            )
        
        # for i in range(len(face_vertices)):
        #     p_start = face_vertices[i]
        #     p_end = face_vertices[(i + 1) % len(face_vertices)]
        #     draw_line(buffer, p_start, p_end, face_vertex_colors[i], face_vertex_colors[(i+1)%len(face_vertices)], width, height)
    return buffer

def get_frame_generator(width, height):
    dz = 1.0
    angle = 0.0
    dt = 1.0 / FPS
    while True:
        yield render_frame_to_tensor(angle, dz, width, height)
        angle += math.pi * dt

def main():
    term_width, term_height = get_terminal_size()
    width, height = term_width, term_height
    
    config = Config(
        width=width, 
        height=height, 
        device=DEVICE, 
        timing_enabled=True,
        timing_file="timing_object.csv",
    )
    
    sys.stdout.buffer.write(ENABLE_ALT_BUFFER + HIDE_CURSOR + CLEAR_SCREEN + b"\033[H")
    sys.stdout.flush()
    
    renderer = AnsiRenderer(frame_generator=get_frame_generator(width, height), config=config)
    
    try:
        for ansi in renderer.get_next_ansi_sequence():
            renderer.render_frame(ansi)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.buffer.write(DISABLE_ALT_BUFFER + SHOW_CURSOR)
        sys.stdout.flush()

if __name__ == "__main__":
    main()

