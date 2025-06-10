"""
rigid_filament3_d_simulation.py
Author: Thomas Quaid
HOOMD‑blue 5.2 prototype – γ‑PFD filament + connector screen
===========================================================

Unit mapping (reduced):
    σ  = 1 nm
    ε  = kBT (300 K)
    m  = 1 (arb.)
Time step:
    dt = 0.005 τ  ≈ 5 fs

This script:
1.  Builds truly rigid filaments (central 'FilCore' + beads).
2.  Builds 3‑patch hubs (central 'HubCore' + three tiny 'P' beads at 120°).
3.  Uses standard LJ potential for orientation‑specific binding between hub patches and filament
    beads.
4.  Runs Brownian dynamics (Langevin) on **cores only**; constituents
    inherit motion via rigid constraints.
5.  Dumps a trajectory (GSD) for post‑processing into GIFs or analysis.

Run 10–20 seeds to build assembly statistics:
    $ python rigid_filament3_d_simulation.py --seed 1
    $ python rigid_filament3_d_simulation.py --seed 2
"""

import argparse
import math
import pathlib
import numpy as np
import hoomd
import hoomd.md

# -------------------------- CLI ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--steps", type=int, default=1_000_000,
                    help="MD steps (dt=0.005)")
args = parser.parse_args()

# ----------------------  parameters  -----------------------
n_filaments      = 8          # cube edges
filament_length  = 6          # beads per filament (inc. 2 ends)
n_connectors     = 12         # cube vertices
box_size         = 300.0      # nm  ( => 300 σ )
spacing          = 1.0        # nm  distance between beads in body frame
hub_patch_r      = 0.5        # nm  arm length to patch bead
kT               = 1.0        # reduced
dt               = 0.005

# Derived counts - CORRECTED: Account for actual particle creation
beads_per_fil    = filament_length - 1           # This should match actual creation
beads_per_hub    = 1 + 3                         # 1 core + 3 patch beads
N_particles      = n_filaments * beads_per_fil + n_connectors * beads_per_hub

# Helper functions
def rand_uvec():
    """Generate a random unit vector"""
    vec = np.random.normal(size=3)
    return vec / np.linalg.norm(vec)

def rotation_matrix_from_axis_angle(axis, angle):
    """
    Create a rotation matrix from axis-angle representation using Rodrigues' formula
    """
    axis = axis / np.linalg.norm(axis)  # normalize axis
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodrigues' rotation formula
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    return R

# ---------------------  device / sim  ----------------------
device = hoomd.device.auto_select()
sim    = hoomd.Simulation(device=device, seed=args.seed)

snap = hoomd.Snapshot()
snap.particles.N = N_particles
# Only define particle types that we actually use
snap.particles.types = ['FilCore', 'F', 'HubCore', 'P']

# Convenience views
pos     = snap.particles.position
typeid  = snap.particles.typeid
mass    = snap.particles.mass
inertia = snap.particles.moment_inertia
body    = snap.particles.body

pid = 0                       # global particle index
body_id = 0                   # body IDs start at 0 (center particles have body = particle_id)

# Store constituent positions for rigid body definition
filament_constituent_pos = []
hub_constituent_pos = []

# Track center particle IDs for proper rigid body setup
filament_center_ids = []
hub_center_ids = []

# ---------------------  filaments  -------------------------
for _ in range(n_filaments):
    # Random centre‑of‑mass within box
    cm  = np.random.uniform(-box_size/4, box_size/4, 3)
    dir = rand_uvec()

    # Core - CRITICAL: Center particle has body = particle_id
    pos[pid]    = cm
    typeid[pid] = 0                       # FilCore
    mass[pid]   = 10.0
    inertia[pid]= (1,1,1)
    body[pid]   = pid                     # FIXED: Center particle body = its own ID
    fil_core_id = pid
    filament_center_ids.append(pid)
    pid += 1

    # Build constituent beads in body frame - FIXED: Create exactly filament_length-2 particles
    constituent_pos = []
    constituent_typ = []

    # CORRECTED: Only create filament_length-2 internal particles (not filament_length-2 + 2)
    for i in range(filament_length-2):          # internal 'F'
        loc = (-((filament_length-1)/2-i-1) * spacing) * np.array([0,0,1])
        constituent_pos.append(loc)
        constituent_typ.append(1)               # 'F'

    # Store for rigid body definition (only need one copy since all filaments are the same)
    if not filament_constituent_pos:
        filament_constituent_pos = constituent_pos.copy()

    # Rotate body‑frame positions into world - FIXED VERSION
    R = rotation_matrix_from_axis_angle(rand_uvec(), np.random.uniform(0, 2*math.pi))
    for loc, typ in zip(constituent_pos, constituent_typ):
        global_pos = cm + R @ loc
        pos[pid]     = global_pos
        typeid[pid]  = typ
        mass[pid]    = 0.0                      # ghost mass
        inertia[pid] = (0,0,0)
        body[pid]    = fil_core_id              # FIXED: Constituents point to center particle ID
        pid += 1

# ---------------------  hubs / connectors ------------------
phi = 2*math.pi/3                                # 120°
patch_positions = [
    ( hub_patch_r, 0.0, 0.0),
    (-hub_patch_r*math.cos(phi/2),  hub_patch_r*math.sin(phi/2), 0.0),
    (-hub_patch_r*math.cos(phi/2), -hub_patch_r*math.sin(phi/2), 0.0),
]

# Store for rigid body definition
hub_constituent_pos = patch_positions.copy()

for _ in range(n_connectors):
    cm = np.random.uniform(-box_size/4, box_size/4, 3)

    # Core - CRITICAL: Center particle has body = particle_id
    pos[pid]    = cm
    typeid[pid] = 2                              # HubCore (index 2 in our types list)
    mass[pid]   = 10.0
    inertia[pid]= (1,1,1)
    body[pid]   = pid                            # FIXED: Center particle body = its own ID
    hub_core_id = pid
    hub_center_ids.append(pid)
    pid += 1

    # Rotate entire hub - FIXED VERSION
    R = rotation_matrix_from_axis_angle(rand_uvec(), np.random.uniform(0, 2*math.pi))

    for loc in patch_positions:
        global_pos = cm + R @ loc
        pos[pid]     = global_pos
        typeid[pid]  = 3                         # 'P' (index 3 in our types list)
        mass[pid]    = 0.0
        inertia[pid] = (0,0,0)
        body[pid]    = hub_core_id               # FIXED: Constituents point to center particle ID
        pid += 1

# DEBUGGING: Check that we used exactly the right number of particles
print(f"Total particles allocated: {N_particles}")
print(f"Particles actually used: {pid}")
assert pid == N_particles, f"Particle count mismatch: expected {N_particles}, used {pid}"

# ---------------------  box / snapshot ---------------------
snap.configuration.box = [box_size, box_size, box_size, 0, 0, 0]
sim.create_state_from_snapshot(snap)

# ------------------  rigid constraints  --------------------
rigid = hoomd.md.constrain.Rigid()
rigid.body['FilCore'] = {
    'constituent_types': ['F'] * (filament_length-2),    # FIXED: Only internal F particles
    'positions': filament_constituent_pos,               # Use stored positions
    'orientations': [(1.0, 0.0, 0.0, 0.0)] * (filament_length-2)  # Add required orientations
}
rigid.body['HubCore'] = {
    'constituent_types': ['P', 'P', 'P'],
    'positions': hub_constituent_pos,
    'orientations': [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]  # Add required orientations
}

# Create integrator and attach rigid constraint first
integrator = hoomd.md.Integrator(dt=dt)
integrator.rigid = rigid

# CRITICAL FIX: Use the special rigid body filter that automatically selects center particles
rigid_centers_filter = hoomd.filter.Rigid(("center",))

integrator.methods.append(
    hoomd.md.methods.Langevin(
        filter=rigid_centers_filter,  # Use the special rigid centers filter
        kT=kT,
        default_gamma=1.0)
)

# ------------------  pair interactions  --------------------
nl = hoomd.md.nlist.Cell(buffer=0.4)

# Define all particle types that we actually use
particle_types = ['FilCore', 'F', 'HubCore', 'P']

# Single LJ potential with all interactions defined
lj = hoomd.md.pair.LJ(nlist=nl)

# Set all interactions systematically
for type1 in particle_types:
    for type2 in particle_types:
        if type1 == 'P' and type2 == 'F':
            # Strong attractive interaction for assembly
            lj.params[(type1, type2)] = dict(epsilon=3.0, sigma=1.2)
            lj.r_cut[(type1, type2)] = 3.0
        elif type1 == 'F' and type2 == 'P':
            # Symmetric interaction
            lj.params[(type1, type2)] = dict(epsilon=3.0, sigma=1.2)
            lj.r_cut[(type1, type2)] = 3.0
        elif type1 in ['FilCore', 'HubCore'] and type2 in ['FilCore', 'HubCore']:
            # Weak repulsion between cores
            lj.params[(type1, type2)] = dict(epsilon=0.1, sigma=2.0)
            lj.r_cut[(type1, type2)] = 2.5
        else:
            # All other interactions: weak excluded volume
            lj.params[(type1, type2)] = dict(epsilon=0.05, sigma=1.0)
            lj.r_cut[(type1, type2)] = 2.0

integrator.forces.append(lj)

sim.operations.integrator = integrator

# ------------------  trajectory writers  -------------------
gsd_dir = pathlib.Path("trajectories")
gsd_dir.mkdir(exist_ok=True)
traj = hoomd.write.GSD(
    filename=str(gsd_dir / f"assembly_seed{args.seed}.gsd"),
    trigger=hoomd.trigger.Periodic(2000),
    mode="wb")
sim.operations.writers.append(traj)

# Simple logging - only basic scalar quantities
log = hoomd.logging.Logger(categories=['scalar'])
log.add(sim, quantities=['timestep', 'walltime'])
table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(10_000), logger=log)
sim.operations.writers.append(table)

# ------------------  run  ----------------------------------
print(f"[seed {args.seed}] running {args.steps} steps …")
sim.run(args.steps)
print("done.  trajectory written to", traj.filename)