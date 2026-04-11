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

def cmap_cycle_highlight(norm):
    """7 — cyan if in a cycle, dark grey if not. norm is binary 0/1."""
    result = np.zeros((len(norm), 4), dtype=np.float32)
    in_c = norm > 0.5
    result[~in_c] = [0.15, 0.15, 0.18, 1.0]   # dim grey — safe node
    result[ in_c] = [0.0,  0.90, 0.85, 1.0]   # vivid cyan — trapped in cycle
    return result

def cmap_flat(norm):
    """8 — every node the same soft blue."""
    return np.tile(np.array([0.45, 0.60, 1.0, 1.0], dtype=np.float32), (len(norm), 1))

def get_colors(metrics, mode):
    norm, cmap, _ = metrics[mode]
    return cmap(norm) if callable(cmap) else vispy_cmap(cmap, norm)

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

    total = len(cycles)
    if total:
        avg     = sum(len(c) for c in cycles) / total
        biggest = len(cycles[0])
    else:
        print("    No dependency cycles found!")
    return cycle_id, cycle_sizes, cycles

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

    cycle_id, cycle_sizes, cycles = compute_scc_data(g)

    # mode 7: 1.0 = in a cycle, 0.0 = not
    in_cycle_norm = (cycle_id >= 0).astype(np.float32)
    # mode 8: uniform value (cmap_flat ignores it)
    flat_norm     = np.full(g.vcount(), 0.5, dtype=np.float32)

    metrics = {
        "pagerank":     (normalize_log(pr),               "plasma",          "PageRank"),
        "in_degree":    (normalize_log(in_deg),           "inferno",         "In-Degree  (dependents)"),
        "out_degree":   (normalize_log(out_deg),          "viridis",         "Out-Degree (dependencies)"),
        "total_degree": (normalize_log(in_deg + out_deg), "magma",           "Total Degree"),
        "betweenness":  (normalize_log(btwn),             cmap_fire,         "Betweenness Centrality"),
        "closeness":    (normalize_linear(clos),          cmap_ice,          "Closeness Centrality"),
        "cycle_hl":     (in_cycle_norm,                   cmap_cycle_highlight, "Cycle Highlight"),
        "flat":         (flat_norm,                       cmap_flat,         "Flat (uniform)"),
    }
    # stash raw SCC arrays for the mouse handler
    metrics["_cycle_id"]    = cycle_id
    metrics["_cycle_sizes"] = cycle_sizes
    metrics["_cycles"]      = cycles
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

    color_state = {"mode": "pagerank"}
    colors      = get_colors(metrics, color_state["mode"])

    # ── Query screen resolution via GLFW before opening the window ────────────
    import glfw as _glfw
    _glfw.init()
    _vm   = _glfw.get_video_mode(_glfw.get_primary_monitor())
    SCR_W = _vm.size.width
    SCR_H = _vm.size.height
    print(f"  screen: {SCR_W}×{SCR_H}")

    # ── Canvas: exact screen size + borderless ────────────────────────────────
    canvas = scene.SceneCanvas(
        keys='interactive', title='Real-Time Physics Graph',
        bgcolor='#0a0a0c', size=(SCR_W, SCR_H), decorate=False, show=True
    )

    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'

    lines = scene.visuals.Line(
        pos=pos, connect=edges, color=(1.0, 1.0, 1.0, 0.05),
        method='gl', parent=view.scene
    )
    lines.order = 0
    lines.set_gl_state(depth_test=False, blend=True)

    markers = scene.visuals.Markers(parent=view.scene)
    markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)
    markers.order = 1
    markers.set_gl_state(depth_test=False, blend=True)

    # ── HUD — change PAD here only ────────────────────────────────────────────
    PAD = 50
    FS  = 13

    # TOP-LEFT: node click info
    info_text = scene.visuals.Text(
        "Click a node for info…", parent=canvas.scene, color='#cccccc',
        pos=(PAD, PAD), font_size=FS,
        anchor_x='left', anchor_y='top'
    )

    # BOTTOM-LEFT: color legend
    color_hud = scene.visuals.Text(
        "", parent=canvas.scene, color='#ffcc00',
        pos=(PAD, SCR_H - 6.5*PAD), font_size=FS,
        anchor_x='left', anchor_y='bottom'
    )

    # TOP-RIGHT: physics controls
    hud_text = scene.visuals.Text(
        "", parent=canvas.scene, color='#00ffcc',
        pos=(SCR_W - PAD, 4*PAD), font_size=FS,
        anchor_x='right', anchor_y='top'
    )

    # ── HUD text builders ─────────────────────────────────────────────────────
    def update_physics_hud():
        state = "RUNNING" if PHYSICS["active"] else "PAUSED"
        mult  = PHYSICS["step_multiplier"]
        hud_text.text = (
            f"Simulation: {state}  [SPACE]\n"
            f"Step mult  [ [ / ] ]: {mult}x\n"
            f"Springs    [ W / S ]: {PHYSICS['spring_strength']:.5f}\n"
            f"Repulsion  [ A / D ]: {PHYSICS['outward_push']:.1f}\n"
            f"Gravity    [ Q / E ]: {PHYSICS['gravity']:.5f}"
        )

    def update_color_hud():
        rows = ["Color mode  [1-8]:"]
        labels = {
            "cycle_hl": "Cycle Highlight  (cyan=in cycle)",
            "flat":     "Flat uniform color",
        }
        for i, key in enumerate(COLOR_KEYS):
            label  = labels.get(key, metrics[key][2])
            marker = ">" if key == color_state["mode"] else " "
            rows.append(f" {marker} {i+1}  {label}")
        color_hud.text = "\n".join(rows)

    update_physics_hud()
    update_color_hud()

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
        lines.set_data(pos=pos)
        markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)

    timer = app.Timer('auto', connect=on_timer_tick, start=True)  # noqa: F841

    # ── Keyboard ──────────────────────────────────────────────────────────────
    @canvas.events.key_press.connect
    def on_key_press(event):
        nonlocal colors
        key_name = event.key.name if hasattr(event, 'key') and hasattr(event.key, 'name') else ""
        key_text = event.text if event.text else ""
        m        = PHYSICS["step_multiplier"]

        if key_name == 'Space' or key_text == ' ':
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
            idx = KEY_BINDS.index(key_text)
            color_state["mode"] = COLOR_KEYS[idx]
            colors = get_colors(metrics, color_state["mode"])
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

            cid   = metrics["_cycle_id"][closest_idx]
            csz   = metrics["_cycle_sizes"][closest_idx]

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

    print("Rendering — enjoy!")
    app.run()


if __name__ == '__main__':
    main()