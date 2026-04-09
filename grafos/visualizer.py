import os
import igraph as ig
import numpy as np
from vispy import app, scene
from vispy.color import get_colormap

app.use_app('glfw') 

# Physics Parameters
PHYSICS = {
    "spring_strength": 0.002,  
    "outward_push": 200.0,    
    "gravity": 0.015,          
    "friction": 0.80,          
    "max_speed": 15.0,         
    "active": False,
    "step_multiplier": 1.0    
}

def main():
    print("Loading graph data...")
    g = ig.Graph.Read_GraphML("pypi_multiseed_10k.graphml")
    
    print(f"Computing initial layout for {g.vcount()} nodes...")
    layout = g.layout_drl()
    
    pos = np.array(layout.coords, dtype=np.float32)
    velocity = np.zeros_like(pos) 
    edges = np.array([e.tuple for e in g.es], dtype=np.uint32)
    
    # For coloring, edit for other analysis
    in_degrees = np.array(g.indegree())
    out_degrees = np.array(g.outdegree())
    
    log_degrees = np.log1p(out_degrees)
    norm_degrees = log_degrees / log_degrees.max()
    sizes = norm_degrees * 21 + 4 
    
    norm_degrees_2d = norm_degrees[:, np.newaxis] 
    
    cmap = get_colormap('plasma')
    colors = cmap.map(norm_degrees)
    #End Coloring
    
    # Initialize Scene
    canvas = scene.SceneCanvas(
        keys='interactive', title='Real-Time Physics Graph', 
        bgcolor='#0a0a0c', size=(1200, 900), show=True
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

    zoom_button_bg = scene.visuals.Rectangle(
        center=(80, 40), width=120, height=35,
        color=(0.2, 0.2, 0.3, 0.9), border_color='#888888',
        parent=canvas.scene
    )
    zoom_button_text = scene.visuals.Text(
        "[ Zoom All (R) ]", parent=canvas.scene, color='white',
        pos=(80, 40), font_size=11, anchor_x='center', anchor_y='center'
    )

    info_panel_bg = scene.visuals.Rectangle(
        center=(220, 100), width=400, height=60,
        color=(0.05, 0.05, 0.08, 0.85), border_color='#444444',
        parent=canvas.scene
    )
    info_text = scene.visuals.Text(
        "Click a node for info...", parent=canvas.scene, color='white',
        pos=(30, 100), font_size=14, anchor_x='left', anchor_y='center'
    )

    physics_panel_bg = scene.visuals.Rectangle(
        center=(170, 840), width=320, height=200,
        color=(0.05, 0.05, 0.08, 0.7), border_color='#333333',
        parent=canvas.scene
    )
    hud_text = scene.visuals.Text(
        "", parent=canvas.scene, color='#00ffcc',
        pos=(30, 840), font_size=12, anchor_x='left', anchor_y='center'
    )

    def update_hud():
        state = "RUNNING" if PHYSICS["active"] else "PAUSED"
        mult = PHYSICS['step_multiplier']
        hud_text.text = (
            f"Simulation: {state} [Press SPACE to toggle]\n"
            f"Step Size Multiplier [[ / ]]: {mult}x\n" # NEW LINE
            f"Springs [W/S]: {PHYSICS['spring_strength']:.5f}\n"
            f"Node Repulsion [A/D]: {PHYSICS['outward_push']:.1f}\n"
            f"Hub Gravity [Q/E]: {PHYSICS['gravity']:.5f}"
        )
    update_hud()

    def on_timer_tick(event):
        if not PHYSICS["active"]:
            return

        nonlocal pos, velocity
        forces = np.zeros_like(pos)

        # Spring Force (Pulls connected nodes together)
        p0 = pos[edges[:, 0]]
        p1 = pos[edges[:, 1]]
        dist_vec = p1 - p0
        spring_f = dist_vec * PHYSICS["spring_strength"]
        
        np.add.at(forces, edges[:, 0], spring_f)
        np.add.at(forces, edges[:, 1], -spring_f)

        # Hub Gravity (Pulls massive packages to the exact center)
        # Multiply by norm_degrees so heavy packages get pulled hard, 
        # but small leaf packages only move because the springs pull them!
        forces -= pos * PHYSICS["gravity"] * norm_degrees_2d

        # TRUE Node-to-Node Repulsion
        # Pick 150 random nodes to act as magnetic repulsors this frame
        sample_size = 300
        repulsor_indices = np.random.choice(len(pos), sample_size, replace=False)
        repulsors = pos[repulsor_indices]
        
        diff = pos[:, np.newaxis, :] - repulsors[np.newaxis, :, :] 
        dist_sq = np.sum(diff**2, axis=2) + 1.0 
        repulsion_force = (diff / dist_sq[:, :, np.newaxis]) * PHYSICS["outward_push"]
        
        # Sum up all the repulsive magnetic forces hitting each node and add it to their momentum
        forces += np.sum(repulsion_force, axis=1)

        # Apply physics and speed limits
        velocity += forces
        velocity *= PHYSICS["friction"] 
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        exceeding = speed > PHYSICS["max_speed"]
        if np.any(exceeding):
            velocity = np.where(exceeding, (velocity / speed) * PHYSICS["max_speed"], velocity)

        pos += np.nan_to_num(velocity)

        # Send to GPU
        lines.set_data(pos=pos)
        markers.set_data(pos=pos, face_color=colors, edge_width=0, size=sizes)

    timer = app.Timer('auto', connect=on_timer_tick, start=True)

    # Keyboard
    @canvas.events.key_press.connect
    def on_key_press(event):
        key_name = event.key.name if hasattr(event, 'key') and hasattr(event.key, 'name') else ""
        key_text = event.text.lower() if event.text else ""
        
        # Grab the current multiplier
        m = PHYSICS["step_multiplier"]

        if key_name == 'Space' or key_text == ' ':
            PHYSICS["active"] = not PHYSICS["active"]
            
        elif key_text == ']':
            PHYSICS["step_multiplier"] *= 10.0
        elif key_text == '[':
            PHYSICS["step_multiplier"] /= 10.0
        elif key_text == 'w':
            PHYSICS["spring_strength"] += (0.001 * m)
        elif key_text == 's':
            PHYSICS["spring_strength"] = max(0.00001, PHYSICS["spring_strength"] - (0.001 * m))
            
        elif key_text == 'd':
            PHYSICS["outward_push"] += (0.5 * m)
        elif key_text == 'a':
            PHYSICS["outward_push"] = max(0.0, PHYSICS["outward_push"] - (0.5 * m))
        elif key_text == 'q':
            PHYSICS["gravity"] += (0.001 * m)
        elif key_text == 'e':
            PHYSICS["gravity"] = max(0.0, PHYSICS["gravity"] - (0.001 * m))
        elif key_text == 'r':
            view.camera.set_range(margin=0.05) 
            
        update_hud()

    # Mouse
    @canvas.events.mouse_press.connect
    def on_mouse_press(event):
        if event.button == 1:
            click_x, click_y = event.pos
            if (20 < click_x < 140) and (22 < click_y < 58):
                view.camera.set_range(margin=0.05)
                return 
            transform = view.scene.transform
            click_data_pos = transform.imap(event.pos)[:2]
            distances = np.linalg.norm(pos - click_data_pos, axis=1)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < 30:
                pkg = g.vs[closest_idx]['id']
                info_text.text = f"Package:  {pkg}\nDependents: {in_degrees[closest_idx]}  |  Required: {out_degrees[closest_idx]}"
            else:
                info_text.text = "Click a node for info..."

    view.camera.set_range(margin=0.05)
    print("Rendering complete.")
    app.run()

if __name__ == '__main__':

    main()