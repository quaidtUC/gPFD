from ovito.io import import_file
from ovito.vis import Viewport
from ovito.pipeline import Pipeline

# Load the GSD file
pipeline = import_file("trajectories/assembly_seed42.gsd")

# Set up visualization
pipeline.modifiers.append(...)  # Add modifiers as needed

# Render frames
vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.render_image(size=(800, 600), filename="ovito_frame.png")