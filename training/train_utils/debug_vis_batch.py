import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2


# ==============================================================================
# Main Visualization Function (Now with robust error handling)
# ==============================================================================
def save_batch_visualization(batch, batch_idx=0, output_base_dir="/home/yijing/workspace/panovggt/output_debug_vis"):
    """
    Saves RGB images, depth maps, and point clouds for a specific item in a batch.
    This function is designed to be robust for debugging: it catches and reports errors
    without crashing the main program or debugging session.

    Args:
        batch (dict): The data batch from your dataloader. Expects keys like 'images', 
                      'depths', 'world_points', 'seq_name'.
        batch_idx (int): The index of the item within the batch to visualize.
        output_base_dir (str): The root directory where output folders will be created.
    """
    try:
        # --- Initial Setup and Validation ---
        batch_size = batch['images'].shape[0]
        if batch_idx >= batch_size:
            print(
                f"[ERROR] Requested batch_idx={batch_idx} is out of range (batch size is {batch_size}). Aborting visualization.")
            return

        # Create a unique output directory based on the sequence name
        seq_name = batch['seq_name'][batch_idx]
        output_dir = os.path.join(output_base_dir, seq_name)

        # Create subdirectories for different data types
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pointclouds"), exist_ok=True)
        print(f"Data will be saved to: {output_dir}")

        # --- Data Extraction and Pre-processing ---
        # Extract the specific item from the batch
        # images: expected shape [Frames, 3, H, W]
        rgb = batch['images'][batch_idx]
        # depths: expected shape [Frames, H, W], unsqueeze to add channel dim
        depth = batch['depths'][batch_idx].unsqueeze(1)
        # world_points: expected shape [Frames, H, W, 3], permute to [Frames, 3, H, W]
        point_map = batch['world_points'][batch_idx].permute(0, 3, 1, 2)

        num_frames = rgb.shape[0]
        H, W = rgb.shape[2], rgb.shape[3]

        print(f"Processing sequence '{seq_name}' with {num_frames} frames...")
        print(f"Image dimensions: {H}x{W}")

        all_points = []
        all_colors = []

        # --- Process Each Frame Individually ---
        for i in range(num_frames):
            print(f"\n--- Processing Frame {i} ---")

            # 1. Save RGB Image
            try:
                rgb_frame = rgb[i].cpu().numpy()  # Shape: [3, H, W]
                rgb_frame_hwc = rgb_frame.transpose(1, 2, 0)  # Shape: [H, W, 3]
                rgb_uint8 = (rgb_frame_hwc * 255).astype(np.uint8)

                rgb_path = os.path.join(output_dir, "rgb", f"frame_{i:03d}.png")
                Image.fromarray(rgb_uint8).save(rgb_path)
                print(f"  [RGB] Saved successfully to: {rgb_path}")
            except Exception as e:
                print(f"  [ERROR] Failed to save RGB image for frame {i}: {e}")
                continue  # Try to process depth and point cloud anyway

            # 2. Save Depth Map
            try:
                depth_frame = depth[i, 0].cpu().numpy()  # Shape: [H, W]

                # Create a visualization of the depth map
                plt.figure(figsize=(10, 5))
                valid_mask_depth = (depth_frame > 0) & np.isfinite(depth_frame)
                vmax = np.percentile(depth_frame[valid_mask_depth], 98) if np.any(valid_mask_depth) else 10.0
                plt.imshow(depth_frame, cmap='viridis', vmin=0, vmax=vmax)
                plt.colorbar(label='Depth (m)')
                plt.title(f'Depth Map - Frame {i}')
                plt.axis('off')
                depth_vis_path = os.path.join(output_dir, "depth", f"frame_{i:03d}_vis.png")
                plt.savefig(depth_vis_path, dpi=150, bbox_inches='tight')
                plt.close()

                # Save raw depth data as 16-bit PNG (in millimeters)
                depth_uint16 = (depth_frame * 1000).astype(np.uint16)
                depth_raw_path = os.path.join(output_dir, "depth", f"frame_{i:03d}_raw.png")
                cv2.imwrite(depth_raw_path, depth_uint16)
                print(f"  [Depth] Visualization and raw data saved.")
            except Exception as e:
                print(f"  [ERROR] Failed to save depth map for frame {i}: {e}")

            # 3. Process Point Cloud
            try:
                points = point_map[i].cpu().numpy().reshape(3, -1).T  # Shape: [H*W, 3]
                colors = rgb_frame_hwc.reshape(-1, 3)  # Shape: [H*W, 3]

                # Filter valid points based on depth
                depth_flat = depth_frame.flatten()
                valid_mask_points = (depth_flat > 0.1) & (depth_flat < 20)  # Filter out very close or far points
                valid_points = points[valid_mask_points]
                valid_colors = colors[valid_mask_points]

                if len(valid_points) > 0:
                    ply_path = os.path.join(output_dir, "pointclouds", f"frame_{i:03d}.ply")
                    save_ply(valid_points, valid_colors, ply_path)
                    print(f"  [Point Cloud] Frame point cloud saved ({len(valid_points)} points).")

                    # Subsample points before appending to the combined list to save memory
                    if len(valid_points) > 50000:
                        indices = np.random.choice(len(valid_points), 50000, replace=False)
                        valid_points = valid_points[indices]
                        valid_colors = valid_colors[indices]
                    all_points.append(valid_points)
                    all_colors.append(valid_colors)
                else:
                    print("  [Point Cloud] No valid points found for this frame.")
            except Exception as e:
                print(f"  [ERROR] Failed to process point cloud for frame {i}: {e}")

        # --- Post-processing and Combined Visualizations ---
        print(f"\n--- Combining and Finalizing Visualizations ---")
        # 4. Save and visualize combined point cloud
        try:
            if all_points:
                all_points = np.vstack(all_points)
                all_colors = np.vstack(all_colors)
                print(f"  [Combined Point Cloud] Total points: {len(all_points)}")

                combined_ply_path = os.path.join(output_dir, "pointclouds", "combined_all_frames.ply")
                save_ply(all_points, all_colors, combined_ply_path)
                print(f"  [Combined Point Cloud] Saved successfully to: {combined_ply_path}")

                visualize_combined_pointcloud(all_points, all_colors, output_dir)
                print("  [Combined Point Cloud] 3D and 2D view images generated.")
            else:
                print("  [Combined Point Cloud] No points to combine.")
        except Exception as e:
            print(f"  [ERROR] Failed to create combined point cloud visualization: {e}")

        # 5. Create panorama montage
        try:
            create_panorama_montage(rgb, depth.squeeze(1), output_dir)
            print("  [Panorama Montage] RGB and Depth montage image generated.")
        except Exception as e:
            print(f"  [ERROR] Failed to create panorama montage: {e}")

    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred in the main visualization function: {e}")
        print("Aborting visualization for this batch.")

    finally:
        print(f"\nVisualization process finished for sequence '{seq_name}'.")


# ==============================================================================
# Helper Functions
# ==============================================================================

def save_ply(points, colors, filename):
    """Saves a point cloud to a PLY file."""
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        # Use a list comprehension for faster writing
        lines = [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n" for p, c in zip(points, colors)]
        f.writelines(lines)


def visualize_combined_pointcloud(points, colors, output_dir):
    """Creates 3D and 2D visualizations of the combined point cloud."""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        # Subsample for faster visualization
        num_vis_points = min(len(points), 100000)
        indices = np.random.choice(len(points), num_vis_points, replace=False)
        vis_points, vis_colors = points[indices], colors[indices]
        if vis_colors.max() > 1.0:
            vis_colors = vis_colors / 255.0

        # 3D View
        fig = plt.figure(figsize=(15, 10));
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], c=vis_colors, s=1, alpha=0.6)
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')
        ax.set_title(f'Combined Point Cloud (Visualizing {num_vis_points} of {len(points)} points)')

        # Equal aspect ratio
        max_range = np.array([vis_points[:, i].max() - vis_points[:, i].min() for i in range(3)]).max() / 2.0
        mid = np.mean(vis_points, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        plt.savefig(os.path.join(output_dir, "combined_pointcloud_3d.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2D Views (Top, Side, Front)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        views = [('Top View (XY)', 0, 1), ('Side View (XZ)', 0, 2), ('Front View (YZ)', 1, 2)]
        for ax_2d, (title, i, j) in zip(axes, views):
            ax_2d.scatter(vis_points[:, i], vis_points[:, j], c=vis_colors, s=1, alpha=0.5)
            ax_2d.set_xlabel(title.split(' ')[2][1]);
            ax_2d.set_ylabel(title.split(' ')[2][2])
            ax_2d.set_title(title);
            ax_2d.axis('equal');
            ax_2d.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_pointcloud_views.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    except ImportError:
        print("[WARNING] mpl_toolkits.mplot3d is not available. Skipping 3D point cloud visualization.")
    except Exception as e:
        # This inner try/except ensures the main function doesn't crash if plotting fails
        print(f"[ERROR] Failed during point cloud plotting: {e}")


def create_panorama_montage(rgb_tensor, depth_tensor, output_dir):
    """Creates a montage of RGB and Depth panoramas."""
    num_frames = rgb_tensor.shape[0]
    fig, axes = plt.subplots(num_frames, 2, figsize=(20, num_frames * 4), squeeze=False)

    for i in range(num_frames):
        # RGB Image
        rgb_frame = rgb_tensor[i].cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        rgb_frame = np.clip(rgb_frame * std + mean, 0, 1).transpose(1, 2, 0)
        axes[i, 0].imshow(rgb_frame)
        axes[i, 0].set_title(f'Frame {i} - RGB')
        axes[i, 0].axis('off')

        # Depth Image
        depth_frame = depth_tensor[i].cpu().numpy()
        valid_mask = (depth_frame > 0) & np.isfinite(depth_frame)
        vmax = np.percentile(depth_frame[valid_mask], 98) if np.any(valid_mask) else 10.0
        im = axes[i, 1].imshow(depth_frame, cmap='viridis', vmin=0, vmax=vmax)
        axes[i, 1].set_title(f'Frame {i} - Depth')
        axes[i, 1].axis('off')
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout(pad=0.5, h_pad=1.5)
    plt.savefig(os.path.join(output_dir, "panorama_montage.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

# ==============================================================================
# HOW TO USE IN YOUR DEBUGGER
# ==============================================================================
# In your debugging console, once the `batch` variable is loaded,
# you can simply call the main function. It will not crash your session.
#
# save_batch_visualization(batch)
#
# # To visualize the second item in the batch (if batch_size > 1)
# save_batch_visualization(batch, batch_idx=2)