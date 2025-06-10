import hoomd
import hoomd.md
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Device
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)

# Parameters
n_filaments = 6
filament_length = 8
n_connectors = 8
spacing = 2.0
box_size = 50.0


def create_snapshot():
    particle_types = ['F', 'C']
    total_particles = n_filaments * filament_length + n_connectors

    # Create snapshot with proper initialization
    snapshot = hoomd.Snapshot()
    snapshot.particles.N = total_particles
    snapshot.particles.types = particle_types

    # Initialize positions and typeid arrays
    positions = []
    typeid = []
    bonds = []  # Store bonds between filament particles

    # Filaments (random start and direction)
    particle_id = 0
    for filament_idx in range(n_filaments):
        start_pos = np.random.uniform(-box_size / 3, box_size / 3, size=3)
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)

        for i in range(filament_length):
            pos = start_pos + i * spacing * direction
            positions.append(pos)
            typeid.append(0)  # 'F'

            # Create bonds between consecutive particles in the filament
            if i > 0:
                bonds.append([particle_id - 1, particle_id])

            particle_id += 1

    # Connectors
    for _ in range(n_connectors):
        pos = np.random.uniform(-box_size / 2, box_size / 2, size=3)
        positions.append(pos)
        typeid.append(1)  # 'C'
        particle_id += 1

    # Convert to numpy arrays with correct shape
    positions = np.array(positions, dtype=np.float64)
    typeid = np.array(typeid, dtype=np.int32)

    # Set particle data
    snapshot.particles.position[:] = positions
    snapshot.particles.typeid[:] = typeid
    snapshot.particles.mass[:] = np.ones(total_particles, dtype=np.float64)
    snapshot.particles.charge[:] = np.zeros(total_particles, dtype=np.float64)
    snapshot.particles.velocity[:] = np.zeros((total_particles, 3), dtype=np.float64)
    snapshot.particles.image[:] = np.zeros((total_particles, 3), dtype=np.int32)

    # Set up bonds
    if bonds:
        snapshot.bonds.N = len(bonds)
        snapshot.bonds.types = ['filament_bond']
        snapshot.bonds.typeid[:] = [0] * len(bonds)
        snapshot.bonds.group[:] = bonds

    # Set box configuration
    snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]

    return snapshot


# Initialize system
snapshot = create_snapshot()
sim.create_state_from_snapshot(snapshot)

# Define bond interactions (this keeps filaments together)
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['filament_bond'] = {'k': 500.0, 'r0': spacing}  # Use dict syntax

# Define pair interactions - CRITICAL FIX: Use proper tuple syntax
nl = hoomd.md.nlist.Cell(buffer=0.4)
pair = hoomd.md.pair.LJ(nlist=nl)

# FIXED: Use tuples (parentheses) for type pairs, not lists
pair.params[('F', 'F')] = {'epsilon': 1.0, 'sigma': 1.0}
pair.params[('F', 'C')] = {'epsilon': 2.0, 'sigma': 1.0}
pair.params[('C', 'C')] = {'epsilon': 0.5, 'sigma': 1.0}

# FIXED: Use tuples for r_cut as well
pair.r_cut[('F', 'F')] = 3.0
pair.r_cut[('F', 'C')] = 3.0
pair.r_cut[('C', 'C')] = 3.0

# Integrator
integrator = hoomd.md.Integrator(dt=0.005)
integrator.forces.append(pair)
integrator.forces.append(harmonic)  # Add bond forces
integrator.methods.append(
    hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.2, default_gamma=0.5)
)
sim.operations.integrator = integrator

# Output
frame_folder = 'frames'
os.makedirs(frame_folder, exist_ok=True)

n_frames = 50  # Reduced for testing
steps_per_frame = 100
filenames = []

print("Starting simulation...")

for frame in range(n_frames):
    sim.run(steps_per_frame)
    snap = sim.state.get_snapshot()
    pos = snap.particles.position
    types = snap.particles.types
    typeid = snap.particles.typeid

    fig = plt.figure(figsize=(8, 8))
    colors = {'F': 'blue', 'C': 'red'}

    # Plot particles
    for p, tid in zip(pos, typeid):
        t = types[tid]
        plt.scatter(p[0], p[1], c=colors[t], s=100, alpha=0.8, edgecolors='k')

    # Draw bonds to visualize filaments
    if hasattr(snap.bonds, 'group') and len(snap.bonds.group) > 0:
        for bond in snap.bonds.group:
            p1, p2 = bond
            x_coords = [pos[p1][0], pos[p2][0]]
            y_coords = [pos[p1][1], pos[p2][1]]
            plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=2)

    plt.xlim(-box_size / 2, box_size / 2)
    plt.ylim(-box_size / 2, box_size / 2)
    plt.axis('off')
    plt.title(f'Frame {frame}: Blue filaments + Red connectors')
    filename = f'{frame_folder}/frame_{frame:03d}.png'
    plt.savefig(filename, dpi=80, bbox_inches='tight')
    plt.close()
    filenames.append(filename)

    if frame % 10 == 0:
        print(f"Frame {frame}/{n_frames} completed")

# Make GIF
print("Creating GIF...")
with imageio.get_writer('cube_self_assembly.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved as 'cube_self_assembly.gif'")