#!/usr/bin/env python3
# visualize.py - Script to visualize VTK output from Rayleigh-Taylor simulation

import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("Error: VTK Python bindings not found. Please install with 'pip install vtk'")
    sys.exit(1)

def read_vtk_file(filename):
    """
    Read a VTK file and extract density, pressure, and velocity data
    """
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    
    data = reader.GetOutput()
    
    # Get dimensions
    dims = data.GetDimensions()
    nx, ny = dims[0], dims[1]
    
    # Extract density
    density_array = data.GetPointData().GetArray("density")
    density = vtk_to_numpy(density_array).reshape((ny, nx))
    
    # Extract pressure if available
    pressure = None
    pressure_array = data.GetPointData().GetArray("pressure")
    if pressure_array:
        pressure = vtk_to_numpy(pressure_array).reshape((ny, nx))
    
    # Extract velocity if available
    velocity = None
    velocity_array = data.GetPointData().GetArray("velocity")
    if velocity_array:
        velocity = vtk_to_numpy(velocity_array).reshape((ny, nx, 3))
    
    return density, pressure, velocity, (nx, ny)

def create_custom_colormap():
    """
    Create a custom colormap for density visualization
    """
    # Colors for our custom colormap
    colors = [(0.05, 0.05, 0.9),    # Light fluid (blue)
              (0.2, 0.7, 1.0),      # Light-medium
              (1.0, 1.0, 1.0),      # Interface
              (1.0, 0.7, 0.2),      # Medium-heavy
              (0.9, 0.05, 0.05)]    # Heavy fluid (red)
    
    return LinearSegmentedColormap.from_list('density_cmap', colors, N=256)

def visualize_density(density, dims, filename, time_step):
    """
    Create and save a visualization of the density field
    """
    plt.figure(figsize=(10, 8))
    
    # Custom colormap
    cmap = create_custom_colormap()
    
    # Plot density
    plt.imshow(density, origin='lower', cmap=cmap, extent=[0, 1, 0, 1], 
               vmin=0.9, vmax=2.1)
    
    plt.colorbar(label='Density')
    plt.title(f'Rayleigh-Taylor Instability - Density - Step {time_step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_velocity(density, velocity, dims, filename, time_step):
    """
    Create and save a visualization of the velocity field
    """
    if velocity is None:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Extract components and downsample for visualization
    nx, ny = dims
    downsample = max(1, nx // 50)  # Downsample to ~50 arrows in each dimension
    
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    U = velocity[::downsample, ::downsample, 0]
    V = velocity[::downsample, ::downsample, 1]
    X = X[::downsample, ::downsample]
    Y = Y[::downsample, ::downsample]
    
    # Plot density as background
    cmap = create_custom_colormap()
    plt.imshow(density, origin='lower', cmap=cmap, extent=[0, 1, 0, 1], 
               vmin=0.9, vmax=2.1, alpha=0.7)
    
    # Plot velocity arrows
    plt.quiver(X, Y, U, V, color='k', scale=15, width=0.002)
    
    plt.colorbar(label='Density')
    plt.title(f'Rayleigh-Taylor Instability - Velocity - Step {time_step}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_time_series_plot(vtk_files, output_filename="velocity_time_series.png"):
    """
    Create a time series plot of maximum velocity magnitude
    """
    max_velocities = []
    time_steps = []
    
    for i, file in enumerate(sorted(vtk_files)):
        _, _, velocity, _ = read_vtk_file(file)
        
        if velocity is not None:
            # Calculate velocity magnitude
            magnitude = np.sqrt(velocity[..., 0]**2 + velocity[..., 1]**2)
            max_velocities.append(np.max(magnitude))
            # Extract time step from filename
            base_filename = os.path.basename(file)
            time_step = int(base_filename.split('_')[1].split('.')[0])
            time_steps.append(time_step)
    
    if max_velocities:
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, max_velocities, 'o-', linewidth=2)
        plt.grid(True)
        plt.title('Maximum Velocity Magnitude vs. Time Step')
        plt.xlabel('Time Step')
        plt.ylabel('Maximum Velocity')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize Rayleigh-Taylor simulation VTK files')
    parser.add_argument('--dir', '-d', default='build', help='Directory containing VTK files (default: build)')
    parser.add_argument('--interval', '-i', type=int, default=1, help='Process every Nth file (default: 1)')
    parser.add_argument('--max', '-m', type=int, default=None, help='Maximum number of files to process (default: all)')
    args = parser.parse_args()
    
    # Get all VTK files from the specified directory
    vtk_dir = args.dir
    vtk_pattern = os.path.join(vtk_dir, 'rt_*.vtk')
    vtk_files = glob.glob(vtk_pattern)
    
    if not vtk_files:
        print(f"No VTK files found in {vtk_dir} using pattern {vtk_pattern}")
        return
    
    # Sort files and apply interval/max filtering
    vtk_files = sorted(vtk_files)
    if args.interval > 1:
        vtk_files = vtk_files[::args.interval]
    if args.max is not None and args.max < len(vtk_files):
        vtk_files = vtk_files[:args.max]
    
    print(f"Found {len(vtk_files)} VTK files in {vtk_dir}. Processing {len(vtk_files)} files.")
    
    # Create output directory for visualizations
    output_dir = os.path.join(vtk_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for file in vtk_files:
        try:
            # Extract time step from filename
            base_filename = os.path.basename(file)
            time_step = int(base_filename.split('_')[1].split('.')[0])
            print(f"Processing {file} (Step {time_step})...")
            
            # Read data
            density, pressure, velocity, dims = read_vtk_file(file)
            
            # Create visualizations
            density_filename = os.path.join(output_dir, f'density_{time_step:04d}.png')
            visualize_density(density, dims, density_filename, time_step)
            
            if velocity is not None:
                velocity_filename = os.path.join(output_dir, f'velocity_{time_step:04d}.png')
                visualize_velocity(density, velocity, dims, velocity_filename, time_step)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Create time series plot
    time_series_filename = os.path.join(output_dir, 'velocity_time_series.png')
    create_time_series_plot(vtk_files, time_series_filename)
    
    print(f"Visualization complete! Results saved in {output_dir}")

if __name__ == "__main__":
    main()