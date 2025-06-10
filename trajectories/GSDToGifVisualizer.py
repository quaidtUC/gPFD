import gsd.hoomd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import os

def visualize_gsd_trajectory(gsd_filename, output_gif="rigid_assembly_from_gsd.gif"):
    """
    Read a .gsd trajectory file and create a 3D animated GIF
    """
    
    try:
        # Open the trajectory
        traj = gsd.hoomd.open(gsd_filename, 'r')
        
        frames_to_plot = min(len(traj), 100)  # Limit to 100 frames for reasonable file size
        step_size = max(1, len(traj) // frames_to_plot)
        
        filenames = []
        
        print(f"Creating visualization from {len(traj)} frames...")
        print(f"Using every {step_size} frame(s)")
        
        # Get particle type mapping
        first_frame = traj[0]
        types = first_frame.particles.types
        print(f"Particle types found: {types}")
        
        # Color scheme
        colors = {
            'FilCore': 'red',
            'F': 'blue', 
            'HubCore': 'green',
            'P': 'gold'
        }
        
        sizes = {
            'FilCore': 150,
            'F': 100,
            'HubCore': 150, 
            'P': 80
        }
        
        # Create temporary frames directory
        temp_dir = 'temp_gsd_frames'
        os.makedirs(temp_dir, exist_ok=True)
        
        frame_count = 0
        for i in range(0, len(traj), step_size):
            frame = traj[i]
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            positions = frame.particles.position
            typeid = frame.particles.typeid
            box = frame.configuration.box
            
            # Count particles by type for debugging
            type_counts = {}
            for tid in typeid:
                particle_type = types[tid]
                type_counts[particle_type] = type_counts.get(particle_type, 0) + 1
            
            # Plot particles
            for pos, tid in zip(positions, typeid):
                particle_type = types[tid]
                color = colors.get(particle_type, 'gray')
                size = sizes.get(particle_type, 50)
                
                ax.scatter(pos[0], pos[1], pos[2], 
                          c=color, s=size, alpha=0.8, 
                          edgecolors='black', linewidth=0.5)
            
            # Set equal aspect ratio and limits
            box_half = box[0] / 2
            ax.set_xlim(-box_half, box_half)
            ax.set_ylim(-box_half, box_half) 
            ax.set_zlim(-box_half, box_half)
            
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            
            # Create title with particle counts
            title = f'Rigid Filament Assembly - Frame {i}\n'
            title += f'Red=FilCore({type_counts.get("FilCore", 0)}), '
            title += f'Blue=F({type_counts.get("F", 0)}), '
            title += f'Green=HubCore({type_counts.get("HubCore", 0)}), '
            title += f'Gold=P({type_counts.get("P", 0)})'
            ax.set_title(title, fontsize=10)
            
            # Rotate view for better perspective
            ax.view_init(elev=20, azim=frame_count*3)
            
            filename = f'{temp_dir}/frame_{frame_count:04d}.png'
            plt.savefig(filename, dpi=80, bbox_inches='tight')
            plt.close()
            filenames.append(filename)
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Create GIF
        print("Creating GIF...")
        with imageio.get_writer(output_gif, mode='I', duration=0.15) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for filename in filenames:
            os.remove(filename)
        os.rmdir(temp_dir)
        
        print(f"✓ GIF saved as: {output_gif}")
        
        # Also create a static overview plot
        create_static_overview(traj, "assembly_overview.png")
        
    except Exception as e:
        print(f"Error processing GSD file: {e}")
        print("Make sure you have gsd package installed: pip install gsd")

def create_static_overview(traj, output_filename):
    """Create a static overview showing first and last frames"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': '3d'})
    
    # First frame
    first_frame = traj[0]
    plot_frame(ax1, first_frame, "Initial Configuration")
    
    # Last frame  
    last_frame = traj[-1]
    plot_frame(ax2, last_frame, "Final Configuration")
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Static overview saved as: {output_filename}")

def plot_frame(ax, frame, title):
    """Helper function to plot a single frame"""
    
    positions = frame.particles.position
    typeid = frame.particles.typeid
    types = frame.particles.types
    box = frame.configuration.box
    
    colors = {'FilCore': 'red', 'F': 'blue', 'HubCore': 'green', 'P': 'gold'}
    sizes = {'FilCore': 150, 'F': 100, 'HubCore': 150, 'P': 80}
    
    for pos, tid in zip(positions, typeid):
        particle_type = types[tid]
        color = colors.get(particle_type, 'gray')
        size = sizes.get(particle_type, 50)
        
        ax.scatter(pos[0], pos[1], pos[2], 
                  c=color, s=size, alpha=0.8, 
                  edgecolors='black', linewidth=0.5)
    
    box_half = box[0] / 2
    ax.set_xlim(-box_half, box_half)
    ax.set_ylim(-box_half, box_half) 
    ax.set_zlim(-box_half, box_half)
    
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_title(title)

# Run the visualization
if __name__ == "__main__":
    print("Converting GSD trajectory to GIF...")
    print("=" * 50)
    
    # Fix the path - since script is IN trajectories folder, file is in same directory
    gsd_file = "assembly_seed42.gsd"  # FIXED: Changed from "trajectories/assembly_seed42.gsd"
    
    if os.path.exists(gsd_file):
        print(f"Found GSD file: {gsd_file}")
        visualize_gsd_trajectory(gsd_file, "rigid_assembly_from_gsd.gif")
    else:
        print(f"GSD file not found: {gsd_file}")
        print("Available files in current directory:")
        for f in os.listdir("."):
            if f.endswith('.gsd'):
                print(f"  - {f}")
        
        # Try to find any .gsd files
        gsd_files = [f for f in os.listdir(".") if f.endswith('.gsd')]
        if gsd_files:
            print(f"\nTrying first available GSD file: {gsd_files[0]}")
            visualize_gsd_trajectory(gsd_files[0], "rigid_assembly_from_gsd.gif")
        else:
            print("No .gsd files found in current directory!")