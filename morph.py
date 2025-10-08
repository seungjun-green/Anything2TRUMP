#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
import imageio
from scipy.spatial import Delaunay
import argparse
import os
from pathlib import Path


def get_image_keypoints(image, num_points=100):
    """Detect keypoints in any image using corner/feature detection"""
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use Shi-Tomasi corner detection
    corners = cv2.goodFeaturesToTrack(gray, num_points, 0.01, 10)
    
    if corners is None or len(corners) < 20:
        # Fallback: create uniform grid if not enough features
        h, w = image.size[1], image.size[0]
        grid_size = int(np.sqrt(num_points))
        x = np.linspace(0, w-1, grid_size)
        y = np.linspace(0, h-1, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    else:
        points = corners.reshape(-1, 2)
    
    # Add border points for complete coverage
    h, w = image.size[1], image.size[0]
    border_points = [
        [0, 0], [w//4, 0], [w//2, 0], [3*w//4, 0], [w-1, 0],
        [0, h//4], [w-1, h//4],
        [0, h//2], [w-1, h//2],
        [0, 3*h//4], [w-1, 3*h//4],
        [0, h-1], [w//4, h-1], [w//2, h-1], [3*w//4, h-1], [w-1, h-1]
    ]
    
    points = np.vstack([points, border_points])
    return points.astype(np.float32)


def apply_affine_transform(src, src_tri, dst_tri, size):
    """Apply affine transform to a triangle"""
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    """Morph one triangle"""
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    t1_rect = []
    t2_rect = []
    t_rect = []
    
    for i in range(3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)
    
    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    
    size = (r[2], r[3])
    
    if size[0] <= 0 or size[1] <= 0:
        return
    
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)
    
    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2
    
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask


def morph_images(img1, img2, points1, points2, alpha):
    """Morph between two images"""
    # Compute intermediate points
    points = (1 - alpha) * points1 + alpha * points2
    
    # Get Delaunay triangulation
    tri = Delaunay(points)
    
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)
    
    for triangle in tri.simplices:
        x, y, z = triangle
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morph, t1, t2, t, alpha)
    
    return img_morph


def create_morph_animation(input_path, target_path, output_path, num_steps=40, hold_frames=10, num_points=150):
    """Create morphing animation from input to target image"""
    
    print(f"Loading images...")
    input_img = Image.open(input_path).convert("RGB").resize((512, 512))
    target_img = Image.open(target_path).convert("RGB").resize((512, 512))
    
    print("Detecting keypoints...")
    input_points = get_image_keypoints(input_img, num_points=num_points)
    target_points = get_image_keypoints(target_img, num_points=num_points)
    
    print(f"Found {len(input_points)} keypoints in input image")
    print(f"Found {len(target_points)} keypoints in target image")
    
    print("Creating morph animation...")
    frames = []
    
    input_array = np.array(input_img)
    target_array = np.array(target_img)
    
    # Hold first image
    for _ in range(hold_frames):
        frames.append(input_array.copy())
    
    # Morph transition
    for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
        print(f"Morphing frame {i+1}/{num_steps} (alpha={alpha:.2f})")
        morphed = morph_images(input_array, target_array, input_points, target_points, alpha)
        frames.append(morphed.astype(np.uint8))
    
    # Hold last image
    for _ in range(hold_frames):
        frames.append(target_array.copy())
    
    print(f"Saving GIF to {output_path}...")
    
    # Create custom duration for each frame
    # First frames (hold_frames): 0.5s each
    # Middle frames (morphing): 0.08s each
    # Last frames (hold_frames): 0.5s each
    durations = [0.5] * hold_frames + [0.08] * num_steps + [0.5] * hold_frames
    
    imageio.mimsave(output_path, frames, duration=durations, loop=0)
    print("Done! Your morphing animation is ready!")


def main():
    parser = argparse.ArgumentParser(description='Morph any image into Trump')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('-o', '--output', type=str, default='output/morph_output.gif', help='Output GIF path (default: output/morph_output.gif)')
    parser.add_argument('-t', '--target', type=str, default='assets/trump.jpg', help='Target image path (default: assets/trump.jpg)')
    parser.add_argument('-s', '--steps', type=int, default=40, help='Number of morphing steps (default: 40)')
    parser.add_argument('-p', '--points', type=int, default=150, help='Number of keypoints (default: 150)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_image):
        print(f"Error: Input image '{args.input_image}' not found!")
        return
    
    # Validate target file
    if not os.path.exists(args.target):
        print(f"Error: Target image '{args.target}' not found!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create morph animation
    create_morph_animation(
        args.input_image,
        args.target,
        args.output,
        num_steps=args.steps,
        num_points=args.points
    )


if __name__ == "__main__":
    main()