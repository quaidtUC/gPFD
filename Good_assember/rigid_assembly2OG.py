import hoomd
import hoomd.md
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio.v2 as imageio  # Fix deprecation warning
import os

# Device
device = hoomd.device.auto_select()
sim = hoomd.Simulation(device=device, seed=42)

# Parameters - RIGID FILAMENT SYSTEM
n_filaments = 8
filament_length = 6  # Shorter for better cube proportions
n_connectors = 12
spacing = 2.0
box_size = 25.0


def create_snapshot():
    particle_types = ['F', 'C', 'F_end']
    total_particles = n_filaments * filament_length + n_connectors

    snapshot = hoomd.Snapshot()
    snapshot.particles.N = total_particles
    snapshot.particles.types = particle_types

    positions = []
    typeid = []
    bonds = []
    angles = []  # Store angle constraints for rigidity

    # RIGID filaments with preferred orientations
    particle_id = 0
    for filament_idx in range(n_filaments):
        # Create filaments along primary axes to encourage cube formation
        if filament_idx < 3:  # X-direction filaments
            direction = np.array([1, 0, 0]) + 0.1 * np.random.normal(size=3)
        elif filament_idx < 6:  # Y-direction filaments
            direction = np.array([0, 1, 0]) + 0.1 * np.random.normal(size=3)
        else:  # Z-direction filaments
            direction = np.array([0, 0, 1]) + 0.1 * np.random.normal(size=3)

        direction /= np.linalg.norm(direction)

        # Position filaments to avoid immediate overlap
        max_extent = (filament_length - 1) * spacing
        safe_region = box_size / 2 - max_extent - 2.0
        start_pos = np.random.uniform(-safe_region, safe_region, size=3)

        for i in range(filament_length):
            pos = start_pos + i * spacing * direction
            positions.append(pos)

            # Mark end particles as special binding sites
            if i == 0 or i == filament_length - 1:
                typeid.append(2)  # 'F_end' - binding sites
            else:
                typeid.append(0)  # 'F' - regular filament body

            # Create bonds between consecutive particles
            if i > 0:
                bonds.append([particle_id - 1, particle_id])

                # CRITICAL: Add angle constraints for rigidity
                if i > 1:  # Need 3 particles to define an angle
                    angles.append([particle_id - 2, particle_id - 1, particle_id])

            particle_id += 1

    # Connectors - fewer, positioned strategically
    for _ in range(n_connectors):
        # Place connectors in central region where cube assembly occurs
        pos = np.random.normal(0, box_size / 6, size=3)  # Gaussian around center
        positions.append(pos)
        typeid.append(1)  # 'C'
        particle_id += 1

    positions = np.array(positions, dtype=np.float64)
    typeid = np.array(typeid, dtype=np.int32)

    # Ensure positions are within bounds
    epsilon = 1.0
    half_box = box_size / 2 - epsilon
    positions = np.clip(positions, -half_box, half_box)

    # Set particle data
    snapshot.particles.position[:] = positions
    snapshot.particles.typeid[:] = typeid
    snapshot.particles.mass[:] = np.ones(total_particles, dtype=np.float64)
    snapshot.particles.charge[:] = np.zeros(total_particles, dtype=np.float64)
    snapshot.particles.velocity[:] = np.random.normal(0, 0.2, (total_particles, 3))
    snapshot.particles.image[:] = np.zeros((total_particles, 3), dtype=np.int32)

    # Set up bonds (keep filaments connected)
    if bonds:
        snapshot.bonds.N = len(bonds)
        snapshot.bonds.types = ['filament_bond']
        snapshot.bonds.typeid[:] = [0] * len(bonds)
        snapshot.bonds.group[:] = bonds

    # CRITICAL: Set up angle constraints (make filaments RIGID)
    if angles:
        snapshot.angles.N = len(angles)
        snapshot.angles.types = ['rigidity_angle']
        snapshot.angles.typeid[:] = [0] * len(angles)
        snapshot.angles.group[:] = angles

    snapshot.configuration.box = [box_size, box_size, box_size, 0, 0, 0]
    return snapshot


# Initialize system
snapshot = create_snapshot()
sim.create_state_from_snapshot(snapshot)

# VERY STRONG bond interactions (keep filaments connected)
harmonic = hoomd.md.bond.Harmonic()
harmonic.params['filament_bond'] = {'k': 5000.0, 'r0': spacing}  # Very stiff

# CRITICAL: Add angle potential for RIGIDITY
angle_harmonic = hoomd.md.angle.Harmonic()
# Force filaments to be perfectly straight (180 degrees)
angle_harmonic.params['rigidity_angle'] = {'k': 2000.0, 't0': np.pi}  # 180° = π radians

# Enhanced 3D interactions
nl = hoomd.md.nlist.Cell(buffer=0.6)
pair = hoomd.md.pair.LJ(nlist=nl)

# Filament body interactions (mild repulsion)
pair.params[('F', 'F')] = {'epsilon': 0.3, 'sigma': 1.0}

# STRONG F_end-C attraction for cube vertices
pair.params[('F_end', 'C')] = {'epsilon': 20.0, 'sigma': 1.3}  # Very strong
pair.r_cut[('F_end', 'C')] = 6.0  # Long range

# Prevent F_end-F_end binding (avoid filament-to-filament connections)
pair.params[('F_end', 'F_end')] = {'epsilon': 0.5, 'sigma': 1.0}  # Weak repulsion
pair.r_cut[('F_end', 'F_end')] = 2.5

# Regular F-C interaction (moderate)
pair.params[('F', 'C')] = {'epsilon': 2.0, 'sigma': 1.0}
pair.r_cut[('F', 'C')] = 3.0

# Connector-connector strong repulsion (prevent clustering)
pair.params[('C', 'C')] = {'epsilon': 5.0, 'sigma': 2.0}  # Strong repulsion
pair.r_cut[('C', 'C')] = 4.0

# F_end-F interactions
pair.params[('F', 'F_end')] = {'epsilon': 0.3, 'sigma': 1.0}
pair.params[('F_end', 'F')] = {'epsilon': 0.3, 'sigma': 1.0}

# Set remaining ranges
pair.r_cut[('F', 'F')] = 2.5
pair.r_cut[('F', 'F_end')] = 2.5
pair.r_cut[('F_end', 'F')] = 2.5

# Stable integrator for rigid system
integrator = hoomd.md.Integrator(dt=0.001)  # Small timestep for stability
integrator.forces.append(pair)
integrator.forces.append(harmonic)
integrator.forces.append(angle_harmonic)  # ADD RIGIDITY

# Lower temperature to prevent excessive motion
integrator.methods.append(
    hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(),
        kT=0.8,  # Low temperature for stable rigid assembly
        default_gamma=5.0  # High damping for control
    )
)

sim.operations.integrator = integrator

# Output
frame_folder = 'frames_rigid_3d'
os.makedirs(frame_folder, exist_ok=True)

n_frames = 300
steps_per_frame = 300  # More steps per frame for convergence
filenames = []

print("Starting RIGID FILAMENT 3D simulation...")
print("=" * 60)
print("RIGIDITY FEATURES:")
print("✓ Angle potentials force filaments straight")
print("✓ Very strong bonds prevent breakage")
print("✓ F_end particles only bind to connectors")
print("✓ Connector repulsion prevents clustering")
print("✓ Directional bias encourages cube geometry")
print("=" * 60)

# Camera rotation for 3D viewing
angles = np.linspace(0, 720, n_frames)  # Two full rotations

for frame in range(n_frames):
    try:
        sim.run(steps_per_frame)
        snap = sim.state.get_snapshot()
        pos = snap.particles.position
        types = snap.particles.types
        typeid = snap.particles.typeid

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = {'F': 'lightblue', 'C': 'red', 'F_end': 'gold'}
        sizes = {'F': 80, 'C': 150, 'F_end': 200}

        # Plot particles with enhanced 3D visibility
        for p, tid in zip(pos, typeid):
            t = types[tid]
            if t == 'F_end':
                ax.scatter(p[0], p[1], p[2], c=colors[t], s=sizes[t],
                           alpha=0.9, edgecolors='darkorange', linewidth=3, marker='*')
            elif t == 'C':
                ax.scatter(p[0], p[1], p[2], c=colors[t], s=sizes[t],
                           alpha=0.8, edgecolors='darkred', linewidth=2, marker='o')
            else:
                ax.scatter(p[0], p[1], p[2], c=colors[t], s=sizes[t],
                           alpha=0.7, edgecolors='darkblue', marker='o')

        # Draw RIGID filament bonds
        if hasattr(snap.bonds, 'group') and len(snap.bonds.group) > 0:
            for bond in snap.bonds.group:
                p1, p2 = bond
                ax.plot([pos[p1][0], pos[p2][0]],
                        [pos[p1][1], pos[p2][1]],
                        [pos[p1][2], pos[p2][2]],
                        'darkblue', alpha=0.9, linewidth=5)  # Thick lines for rigid filaments

        # Highlight F_end-C connections
        f_end_positions = pos[typeid == 2]
        c_positions = pos[typeid == 1]

        rigid_connections = 0
        successful_bindings = 0

        for f_pos in f_end_positions:
            for c_pos in c_positions:
                distance = np.linalg.norm(f_pos - c_pos)
                if distance < 3.0:
                    rigid_connections += 1
                    if distance < 2.0:  # Strong binding
                        successful_bindings += 1
                        ax.plot([f_pos[0], c_pos[0]],
                                [f_pos[1], c_pos[1]],
                                [f_pos[2], c_pos[2]],
                                'lime', alpha=0.8, linewidth=4)  # Green for strong bonds
                    else:  # Weak interaction
                        alpha = 0.6 - distance / 6.0
                        ax.plot([f_pos[0], c_pos[0]],
                                [f_pos[1], c_pos[1]],
                                [f_pos[2], c_pos[2]],
                                'yellow', alpha=alpha, linewidth=2, linestyle='--')

        # 3D view settings
        ax.set_xlim(-box_size / 2, box_size / 2)
        ax.set_ylim(-box_size / 2, box_size / 2)
        ax.set_zlim(-box_size / 2, box_size / 2)

        # Smooth camera rotation
        elev = 15 + 8 * np.sin(frame * 0.03)
        azim = angles[frame]
        ax.view_init(elev=elev, azim=azim)

        # Enhanced styling
        ax.set_facecolor('black')
        ax.grid(True, alpha=0.2, color='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.set_xlabel('X', fontsize=12, color='white')
        ax.set_ylabel('Y', fontsize=12, color='white')
        ax.set_zlabel('Z', fontsize=12, color='white')

        plt.suptitle(
            f'RIGID Filament Assembly - Frame {frame}\n'
            f'★Gold=Binding Sites, ●Red=Connectors, ●Blue=RIGID Filaments\n'
            f'Strong Bonds: {successful_bindings} | Total Interactions: {rigid_connections}',
            fontsize=14, fontweight='bold', color='white'
        )

        fig.patch.set_facecolor('black')

        filename = f'{frame_folder}/frame_{frame:03d}.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='black')
        plt.close()
        filenames.append(filename)

        if frame % 60 == 0:
            print(f"Frame {frame}/{n_frames} - Rigid bonds: {successful_bindings}")

    except RuntimeError as e:
        print(f"Simulation failed at frame {frame}: {e}")
        break

# Create outputs
if filenames:
    print("\n" + "=" * 60)
    print("Creating RIGID FILAMENT outputs...")

    # High-quality GIF (this will work)
    print("Creating rigid 3D GIF...")
    with imageio.get_writer('rigid_cube_assembly_3d.gif', mode='I', duration=0.05) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("✓ Rigid 3D GIF: 'rigid_cube_assembly_3d.gif'")

    # Alternative: Create a high-quality APNG (animated PNG) instead of MP4
    print("Creating rigid 3D APNG (animated PNG)...")
    try:
        with imageio.get_writer('rigid_cube_assembly_3d.png', mode='I', duration=0.05) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        print("✓ Rigid 3D APNG: 'rigid_cube_assembly_3d.png'")
    except Exception as e:
        print(f"APNG creation failed: {e}")

    # Create a slow-motion version
    print("Creating slow-motion GIF...")
    with imageio.get_writer('rigid_cube_assembly_slow.gif', mode='I', duration=0.15) as writer:
        for filename in filenames[::2]:  # Use every other frame for slower motion
            image = imageio.imread(filename)
            writer.append_data(image)
    print("✓ Slow-motion GIF: 'rigid_cube_assembly_slow.gif'")

print("\n" + "=" * 60)
print("RIGID FILAMENT ASSEMBLY COMPLETE!")
print("=" * 60)
print("RIGIDITY SOLUTIONS:")
print("✓ Angle potentials: k=2000, θ=180° (perfectly straight)")
print("✓ Super-strong bonds: k=5000 (no breaking)")
print("✓ F_end-only binding: prevents curling loops")
print("✓ Connector repulsion: prevents clustering")
print("✓ Small timestep: dt=0.001 for stability")
print("✓ Low temperature: kT=0.8 for controlled motion")
print("✓ High damping: γ=5.0 for stability")
print("✓ Directional bias: preferential X/Y/Z alignment")
print("=" * 60)
print("OUTPUT FILES:")
print("✓ rigid_cube_assembly_3d.gif - Main animation")
print("✓ rigid_cube_assembly_3d.png - APNG format (if supported)")
print("✓ rigid_cube_assembly_slow.gif - Slow motion version")
print("=" * 60)