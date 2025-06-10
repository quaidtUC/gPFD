import hoomd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Parameters
n_particles = 30
spacing = 1.5
box_size = n_particles * spacing
filament_length = 10

def make_filament():
    positions = []
    for i in range(n_particles):
        x = i * spacing
        positions.append([x, 0, 0])
    return np.array(positions)

positions = make_filament()

# Create device and simulation (HOOMD 5.2 style)
device = hoomd.device.CPU()
simulation = hoomd.Simulation(device=device, seed=42)

# Create snapshot using HOOMD 5.2 API - Fixed approach
snapshot = hoomd.Snapshot()
snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]

# CRITICAL FIX: Set N first, then resize arrays
snapshot.particles.N = n_particles
snapshot.particles.types = ['A']

# Now resize the arrays properly before assignment
snapshot.particles.resize(n_particles)

# Initialize arrays properly - now they are correctly sized
snapshot.particles.typeid[:] = [0] * n_particles
snapshot.particles.position[:] = positions
snapshot.particles.mass[:] = [1.0] * n_particles
snapshot.particles.charge[:] = [0.0] * n_particles

# Create state from snapshot
simulation.create_state_from_snapshot(snapshot)

# Create neighbor list and pair potential
nl = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=nl)
lj.params[('A', 'A')] = {'epsilon': 1.0, 'sigma': 1.0}
lj.r_cut[('A', 'A')] = 2.5

# Create integrator
integrator = hoomd.md.Integrator(dt=0.005)
integrator.forces.append(lj)

# Add NVT method (Langevin thermostat in HOOMD 5.2)
langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.2, default_gamma=0.5)
integrator.methods.append(langevin)

# Set integrator
simulation.operations.integrator = integrator

# Output folder for frames
frame_folder = 'frames'
os.makedirs(frame_folder, exist_ok=True)

# Run and save frames for GIF
n_frames = 100
steps_per_frame = 100
filenames = []

for frame in range(n_frames):
    simulation.run(steps_per_frame)
    
    # Get positions from snapshot
    snapshot = simulation.state.get_snapshot()
    pos = snapshot.particles.position

    fig = plt.figure(figsize=(8, 2))
    plt.scatter(pos[:, 0], pos[:, 1], c='blue', s=80)
    plt.xlim(-1, box_size + 1)
    plt.ylim(-5, 5)
    plt.axis('off')
    filename = f'{frame_folder}/frame_{frame:03d}.png'
    plt.savefig(filename, dpi=80)
    plt.close()
    filenames.append(filename)

# Create GIF
with imageio.get_writer('filament_motion.gif', mode='I', duration=0.05) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved as 'filament_motion.gif'")