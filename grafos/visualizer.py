import colorsys

import igraph as ig
import numpy as np
from vispy import app, scene
from vispy.color import get_colormap

app.use_app('glfw')

# ── Physics Parameters ────────────────────────────────────────────────────────
PHYSICS = {
    "spring_strength": 0.002,
    "outward_push":    200.0,
    "gravity":         0.015,
    "friction":        0.80,
    "max_speed":       15.0,
    "active":          False,
    "step_multiplier": 1.0,
}

# ── Coloring helpers ──────────────────────────────────────────────────────────
def normalize_log(arr):
    arr = np.array(arr, dtype=np.float64)
    arr = np.log1p(np.clip(arr, 0, None))
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

def normalize_linear(arr):
    arr = np.array(arr, dtype=np.float64)
    rng = arr.max() - arr.min()
    return (arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)

def gradient_colors(norm, color_stops):
    norm  = np.clip(norm, 0.0, 1.0).astype(np.float32)
    stops = np.array(color_stops, dtype=np.float32)
    n_seg = len(stops) - 1
    t     = norm * n_seg
    idx   = np.clip(t.astype(int), 0, n_seg - 1)
    frac  = (t - idx)[:, np.newaxis]
    rgb   = stops[idx] * (1 - frac) + stops[idx + 1] * frac
    alpha = np.ones((len(norm), 1), dtype=np.float32)
    return np.concatenate([rgb, alpha], axis=1)

def vispy_cmap(name, norm):
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    return get_colormap(name).map(norm).astype(np.float32)

def cmap_fire(norm):
    return gradient_colors(norm, [
        (0.0, 0.0, 0.0), (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (1.0, 1.0, 1.0),
    ])

def cmap_ice(norm):
    return gradient_colors(norm, [
        (0.0, 0.0, 0.1), (0.0, 0.3, 0.6),
        (0.0, 0.7, 1.0), (0.8, 1.0, 1.0),
    ])

def cmap_flat(norm):
    """8 — every node the same soft blue."""
    return np.tile(np.array([0.45, 0.60, 1.0, 1.0], dtype=np.float32), (len(norm), 1))

def get_colors(metrics, mode):
    norm, cmap, _ = metrics[mode]
    return cmap(norm) if callable(cmap) else vispy_cmap(cmap, norm)

# ── Cycle palette & per-cycle node colors ─────────────────────────────────────
def generate_cycle_palette(n_cycles, saturation=0.82, value=0.95):
    """Evenly spaced hues across the HSV wheel for n_cycles distinct colors."""
    if n_cycles == 0:
        return np.zeros((1, 4), dtype=np.float32)
    palette = np.zeros((n_cycles, 4), dtype=np.float32)
    for i in range(n_cycles):
        h = i / n_cycles
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        palette[i] = [r, g, b, 1.0]
    return palette

def node_colors_mode7(cycle_id, cycle_palette):
    """Per-node RGBA: dim grey if not in any cycle, else unique cycle hue."""
    n   = len(cycle_id)
    out = np.tile(np.array([0.10, 0.10, 0.12, 1.0], dtype=np.float32), (n, 1))
    mask = cycle_id >= 0
    if mask.any():
        out[mask] = cycle_palette[cycle_id[mask] % len(cycle_palette)]
    return out

def line_vertex_colors_mode7(pos_len, cycle_id, cycle_palette):
    """
    Per-vertex color array for the line visual (one RGBA per node position).
    Cycle nodes get their cycle hue; non-cycle nodes get transparent dark grey
    so their attached lines fade to invisible at the non-cycle end.
    """
    out = np.tile(np.array([0.10, 0.10, 0.12, 0.0], dtype=np.float32), (pos_len, 1))
    mask = cycle_id >= 0
    if mask.any():
        c = cycle_palette[cycle_id[mask] % len(cycle_palette)].copy()
        c[:, 3] = 0.35          # intra-cycle edge alpha
        out[mask] = c
    return out

# ── SCC cycle computation ─────────────────────────────────────────────────────
def compute_scc_data(g):
    print("  · strongly connected components …")
    scc         = g.connected_components(mode="strong")
    cycle_id    = np.full(g.vcount(), -1, dtype=np.int32)
    cycle_sizes = np.zeros(g.vcount(), dtype=np.int32)
    cycles      = sorted([list(c) for c in scc if len(c) > 1], key=len, reverse=True)

    for cid, members in enumerate(cycles):
        for v in members:
            cycle_id[v]    = cid
            cycle_sizes[v] = len(members)

    n_cycles = len(cycles)
    if n_cycles:
        avg     = sum(len(c) for c in cycles) / n_cycles
        biggest = len(cycles[0])
        print(f"    {n_cycles} cycles found — biggest={biggest}, avg={avg:.1f}")
    else:
        print("    No dependency cycles found!")

    cycle_palette = generate_cycle_palette(n_cycles)
    return cycle_id, cycle_sizes, cycles, cycle_palette

# ── Edge masks for mode 7 ─────────────────────────────────────────────────────
def compute_mode7_edge_masks(edges, cycle_id):
    """
    Partition all edges into three groups for mode-7 rendering:
      intra  — both endpoints in the SAME cycle   → coloured by cycle hue
      inter  — endpoints in DIFFERENT cycles       → bright bridge lines
      other  — at least one endpoint not in cycle  → hidden in mode 7

    Note: a true "shortest path visiting all cycles" (Steiner / TSP) is
    NP-hard and infeasible for graphs with thousands of cycles or a single
    SCC of 3 500+ nodes. Showing the raw inter-cycle edges in the graph
    gives the same spatial insight without the combinatorial explosion.
    """
    src_cid = cycle_id[edges[:, 0]]
    tgt_cid = cycle_id[edges[:, 1]]

    both_in_cycle = (src_cid >= 0) & (tgt_cid >= 0)
    intra_mask    = both_in_cycle & (src_cid == tgt_cid)
    inter_mask    = both_in_cycle & (src_cid != tgt_cid)

    intra_edges = edges[intra_mask]
    inter_edges = edges[inter_mask]
    print(f"    edge split — intra-cycle={intra_mask.sum():,}  "
          f"inter-cycle={inter_mask.sum():,}  "
          f"hidden={((~intra_mask) & (~inter_mask)).sum():,}")
    return intra_edges, inter_edges

# ── Metric precomputation ─────────────────────────────────────────────────────
def precompute_metrics(g):
    print("  · degree …")
    in_deg  = np.array(g.indegree(),  dtype=np.float64)
    out_deg = np.array(g.outdegree(), dtype=np.float64)
    print("  · pagerank …")
    pr = np.array(g.pagerank(damping=0.85))
    print("  · betweenness …")
    btwn = np.array(g.betweenness(directed=True))
    print("  · closeness …")
    clos = np.nan_to_num(np.array(g.closeness()), nan=0.0)

    cycle_id, cycle_sizes, cycles, cycle_palette = compute_scc_data(g)

    # mode 7: dummy norm (coloring is handled separately via cycle_palette)
    in_cycle_norm = (cycle_id >= 0).astype(np.float32)
    # mode 8: uniform value (cmap_flat ignores it)
    flat_norm     = np.full(g.vcount(), 0.5, dtype=np.float32)

    # cmap_cycle_highlight is now a no-op placeholder;
    # actual colours are computed in node_colors_mode7()
    def cmap_cycle_highlight(norm):
        return node_colors_mode7(cycle_id, cycle_palette)

    metrics = {
        "pagerank":     (normalize_log(pr),               "plasma",          "PageRank"),
        "in_degree":    (normalize_log(in_deg),           "inferno",         "In-Degree  (dependents)"),
        "out_degree":   (normalize_log(out_deg),          "viridis",         "Out-Degree (dependencies)"),
        "total_degree": (normalize_log(in_deg + out_deg), "magma",           "Total Degree"),
        "betweenness":  (normalize_log(btwn),             cmap_fire,         "Betweenness Centrality"),
        "closeness":    (normalize_linear(clos),          cmap_ice,          "Closeness Centrality"),
        "cycle_hl":     (in_cycle_norm,                   cmap_cycle_highlight, "Cycle Highlight (per-cycle color)"),
        "flat":         (flat_norm,                       cmap_flat,         "Flat (uniform)"),
    }
    metrics["_cycle_id"]      = cycle_id
    metrics["_cycle_sizes"]   = cycle_sizes
    metrics["_cycles"]        = cycles
    metrics["_cycle_palette"] = cycle_palette
    return metrics

COLOR_KEYS = [
    "pagerank", "in_degree", "out_degree", "total_degree",
    "betweenness", "closeness", "cycle_hl", "flat",
]
KEY_BINDS = list("12345678")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading graph data…")
    g = ig.Graph.Read_GraphML("pypi_multiseed_10k.graphml")
    print(f"  {g.vcount()} nodes, {g.ecount()} edges")

    print("Computing layout…")
    layout   = g.layout_drl()
    pos      = np.array(layout.coords, dtype=np.float32)
    velocity = np.zeros_like(pos)
    edges    = np.array([e.tuple for e in g.es], dtype=np.uint32)

    in_deg  = np.array(g.indegree(),  dtype=np.float64)
    out_deg = np.array(g.outdegree(), dtype=np.float64)
    log_deg         = np.log1p(out_deg + in_deg)
    norm_degrees    = log_deg / log_deg.max()
    norm_degrees_2d = norm_degrees[:, np.newaxis]
    sizes           = norm_degrees * 21 + 4

    print("Precomputing centrality metrics…")
    metrics = precompute_metrics(g)

    cycle_id      = metrics["_cycle_id"]
    cycle_palette = metrics["_cycle_palette"]

    print("Precomputing mode-7 edge partitions…")
    intra_edges, inter_edges = compute_mode7_edge_masks(edges, cycle_id)

    color_state  = {"mode": "pagerank"}
    invert_state = {"active": False}
    colors       = get_colors(metrics, color_state["mode"])

    # ── Query screen resolution via GLFW before opening the window ────────────
    import glfw as _glfw
    _glfw.init()
    _vm   = _glfw.get_video_mode(_glfw.get_primary_monitor())
    SCR_W = _vm.size.width
    SCR_H = _vm.size.height
    print(f"  screen: {SCR_W}×{SCR_H}")

    # ── Canvas ────────────────────────────────────────────────────────────────
    canvas = scene.SceneCanvas(
        keys='interactive', title='Real-Time Physics Graph',
        bgcolor='#0a0a0c', size=(SCR_W, SCR_H), decorate=False, show=True
    )

    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    # ── Line visuals (three layers) ───────────────────────────────────────────
    # Layer 0: all edges — used in every mode EXCEPT mode 7
    lines_all = scene.visuals.Line(
        pos=pos, connect=edges, color=(1.0, 1.0, 1.0, 0.05),
        method='gl', parent=view.scene
    )
    lines_all.order = 0
    lines_all.set_gl_state(depth_test=False, blend=True)

    # Layer 1: intra-cycle edges — only visible in mode 7
    # Vertex colours: each node gets its cycle hue so edges blend between them.
    _intra_vc = line_vertex_colors_mode7(len(pos), cycle_id, cycle_palette)
    lines_intra = scene.visuals.Line(
        pos=pos,
        connect=intra_edges if len(intra_edges) else np.zeros((0, 2), dtype=np.uint32),
        color=_intra_vc,
        method='gl', parent=view.scene
    )
    lines_intra.order = 1
    lines_intra.set_gl_state(depth_test=False, blend=True)
    lines_intra.visible = False

    # Layer 2: inter-cycle bridge edges — only visible in mode 7
    # Bright white/gold so cross-cycle bridges stand out clearly.
    lines_inter = scene.visuals.Line(
        pos=pos,
        connect=inter_edges if len(inter_edges) else np.zeros((0, 2), dtype=np.uint32),
        color=(1.0, 0.85, 0.20, 0.55),   # warm gold, semi-transparent
        method='gl', parent=view.scene
    )
    lines_inter.order = 2
    lines_inter.set_gl_state(depth_test=False, blend=True)
    lines_inter.visible = False

    # ── Markers ───────────────────────────────────────────────────────────────
    markers = scene.visuals.Markers(parent=view.scene)
    markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)
    markers.order = 3
    markers.set_gl_state(depth_test=False, blend=True)

    # ── HUD ───────────────────────────────────────────────────────────────────
    PAD = 50
    FS  = 13

    info_text = scene.visuals.Text(
        "Click a node for info…", parent=canvas.scene, color='#cccccc',
        pos=(PAD, PAD), font_size=FS,
        anchor_x='left', anchor_y='top'
    )

    color_hud = scene.visuals.Text(
        "", parent=canvas.scene, color='#ffcc00',
        pos=(PAD, SCR_H - 7*PAD), font_size=FS,
        anchor_x='left', anchor_y='bottom'
    )

    hud_text = scene.visuals.Text(
        "", parent=canvas.scene, color='#00ffcc',
        pos=(SCR_W - PAD, 4*PAD), font_size=FS,
        anchor_x='right', anchor_y='top'
    )

    # ── Helper: switch edge-layer visibility ──────────────────────────────────
    def set_mode7_edges(active):
        lines_all.visible   = not active
        lines_intra.visible = active
        lines_inter.visible = active

    # ── HUD builders ──────────────────────────────────────────────────────────
    def update_physics_hud():
        state = "RUNNING" if PHYSICS["active"] else "PAUSED"
        mult  = PHYSICS["step_multiplier"]
        hud_text.text = (
            f"Simulation: {state}  [SPACE]\n"
            f"Invert colors  [ I ]\n"
            f"Step mult  [ [ / ] ]: {mult}x\n"
            f"Springs    [ W / S ]: {PHYSICS['spring_strength']:.5f}\n"
            f"Repulsion  [ A / D ]: {PHYSICS['outward_push']:.1f}\n"
            f"Gravity    [ Q / E ]: {PHYSICS['gravity']:.5f}"
        )

    def update_color_hud():
        n_cycles = len(metrics["_cycles"])
        rows = [
            "Color mode  [1-8]:",
        ]
        labels = {
            "cycle_hl": "Cycle Highlight  (per-cycle color + bridge edges)",
            "flat":     "Flat uniform color",
        }
        for i, key in enumerate(COLOR_KEYS):
            label  = labels.get(key, metrics[key][2])
            marker = ">" if key == color_state["mode"] else " "
            rows.append(f" {marker} {i+1}  {label}")
        color_hud.text = "\n".join(rows)

    update_physics_hud()
    update_color_hud()

    # ── Invert helpers ────────────────────────────────────────────────────────
    def apply_invert(rgba):
        inv = rgba.copy()
        inv[:, :3] = 1.0 - inv[:, :3]
        return inv

    def refresh_after_invert():
        nonlocal colors
        c = get_colors(metrics, color_state["mode"])
        if invert_state["active"]:
            c = apply_invert(c)
        colors = c
        markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)

        in_mode7 = color_state["mode"] == "cycle_hl"
        if in_mode7:
            _vc = line_vertex_colors_mode7(len(pos), cycle_id, cycle_palette)
            if invert_state["active"]:
                _vc = apply_invert(_vc)
            lines_intra.set_data(pos=pos, color=_vc)
            inter_col = (0.20, 0.35, 1.0, 0.55) if invert_state["active"] else (1.0, 0.85, 0.20, 0.55)
            lines_inter.set_data(pos=pos, color=inter_col)
        else:
            edge_col = (0.0, 0.0, 0.0, 0.15) if invert_state["active"] else (1.0, 1.0, 1.0, 0.05)
            lines_all.set_data(pos=pos, color=edge_col)

        canvas.bgcolor  = '#f5f5f3' if invert_state["active"] else '#0a0a0c'
        info_text.color = '#222222' if invert_state["active"] else '#cccccc'
        color_hud.color = '#884400' if invert_state["active"] else '#ffcc00'
        hud_text.color  = '#006644' if invert_state["active"] else '#00ffcc'
        canvas.update()

    # ── Timer ─────────────────────────────────────────────────────────────────
    first_tick = [True]

    def on_timer_tick(event):
        if first_tick[0]:
            try:
                view.camera.set_range(margin=0.05)
            except Exception:
                pass
            first_tick[0] = False

        if not PHYSICS["active"]:
            return
        nonlocal pos, velocity

        forces = np.zeros_like(pos)
        p0 = pos[edges[:, 0]]
        p1 = pos[edges[:, 1]]
        spring_f = (p1 - p0) * PHYSICS["spring_strength"]
        np.add.at(forces, edges[:, 0],  spring_f)
        np.add.at(forces, edges[:, 1], -spring_f)

        forces -= pos * PHYSICS["gravity"] * norm_degrees_2d

        sample_size = 300
        rep_idx   = np.random.choice(len(pos), sample_size, replace=False)
        repulsors = pos[rep_idx]
        diff      = pos[:, np.newaxis, :] - repulsors[np.newaxis, :, :]
        dist_sq   = np.sum(diff**2, axis=2) + 1.0
        rep_force = (diff / dist_sq[:, :, np.newaxis]) * PHYSICS["outward_push"]
        forces   += np.sum(rep_force, axis=1)

        velocity += forces
        velocity *= PHYSICS["friction"]
        speed     = np.linalg.norm(velocity, axis=1, keepdims=True)
        fast      = speed > PHYSICS["max_speed"]
        if np.any(fast):
            velocity = np.where(fast, (velocity / speed) * PHYSICS["max_speed"], velocity)

        pos += np.nan_to_num(velocity)

        in_mode7 = color_state["mode"] == "cycle_hl"

        # Update the active line layer(s)
        if in_mode7:
            _vc = line_vertex_colors_mode7(len(pos), cycle_id, cycle_palette)
            lines_intra.set_data(pos=pos, color=_vc)
            lines_inter.set_data(pos=pos)
        else:
            lines_all.set_data(pos=pos)

        markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)

    timer = app.Timer('auto', connect=on_timer_tick, start=True)  # noqa: F841

    # ── Keyboard ──────────────────────────────────────────────────────────────
    @canvas.events.key_press.connect
    def on_key_press(event):
        nonlocal colors
        key_name = event.key.name if hasattr(event, 'key') and hasattr(event.key, 'name') else ""
        key_text = event.text if event.text else ""
        m        = PHYSICS["step_multiplier"]

        if key_text.lower() == 'i':
            invert_state["active"] = not invert_state["active"]
            refresh_after_invert()
            return
        elif key_name == 'Space' or key_text == ' ':
            PHYSICS["active"] = not PHYSICS["active"]
        elif key_text == ']':
            PHYSICS["step_multiplier"] *= 10.0
        elif key_text == '[':
            PHYSICS["step_multiplier"] /= 10.0
        elif key_text.lower() == 'w':
            PHYSICS["spring_strength"] += 0.001 * m
        elif key_text.lower() == 's':
            PHYSICS["spring_strength"] = max(0.00001, PHYSICS["spring_strength"] - 0.001 * m)
        elif key_text.lower() == 'd':
            PHYSICS["outward_push"] += 0.5 * m
        elif key_text.lower() == 'a':
            PHYSICS["outward_push"] = max(0.0, PHYSICS["outward_push"] - 0.5 * m)
        elif key_text.lower() == 'q':
            PHYSICS["gravity"] += 0.001 * m
        elif key_text.lower() == 'e':
            PHYSICS["gravity"] = max(0.0, PHYSICS["gravity"] - 0.001 * m)
        elif key_text in KEY_BINDS:
            idx  = KEY_BINDS.index(key_text)
            new_mode = COLOR_KEYS[idx]
            entering_mode7 = new_mode == "cycle_hl"
            leaving_mode7  = color_state["mode"] == "cycle_hl" and not entering_mode7

            color_state["mode"] = new_mode
            colors = get_colors(metrics, color_state["mode"])

            # Swap edge-layer visibility when crossing the mode-7 boundary
            if entering_mode7 or leaving_mode7:
                set_mode7_edges(entering_mode7)
                if entering_mode7:
                    # Sync intra-cycle vertex colours to current positions
                    _vc = line_vertex_colors_mode7(len(pos), cycle_id, cycle_palette)
                    lines_intra.set_data(pos=pos, color=_vc)
                    lines_inter.set_data(pos=pos)
                else:
                    # lines_all may have drifted while physics ran in mode 7
                    lines_all.set_data(pos=pos)
            canvas.update()

            markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)
            update_color_hud()
            return

        update_physics_hud()

    # ── Mouse ─────────────────────────────────────────────────────────────────
    @canvas.events.mouse_press.connect
    def on_mouse_press(event):
        if event.button != 1:
            return
        transform      = view.scene.transform
        click_data_pos = transform.imap(event.pos)[:2]
        distances      = np.linalg.norm(pos - click_data_pos, axis=1)
        closest_idx    = np.argmin(distances)

        if distances[closest_idx] < 30:
            pkg        = g.vs[closest_idx]['id']
            mode_label = metrics[color_state["mode"]][2]
            score      = metrics[color_state["mode"]][0][closest_idx]

            cid = metrics["_cycle_id"][closest_idx]
            csz = metrics["_cycle_sizes"][closest_idx]

            if cid >= 0:
                cycle_info = f"  |  cycle #{cid}  size={csz}"
            else:
                cycle_info = "  |  no cycle"

            info_text.text = (
                f"{pkg}   "
                f"in={int(in_deg[closest_idx])}  "
                f"out={int(out_deg[closest_idx])}  "
                f"{mode_label}: {score:.4f}"
                f"{cycle_info}"
            )
        else:
            info_text.text = "Click a node for info…"

    app.run()


if __name__ == '__main__':
    main()