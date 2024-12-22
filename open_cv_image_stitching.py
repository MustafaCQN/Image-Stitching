import cv2
import numpy as np
import os
from pathlib import Path

def stitch_images(image_paths):
    # Read all images
    images = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    
    if not images:
        print("No valid images found")
        return None

    # Create a stitcher object
    stitcher = cv2.Stitcher_create(mode=cv2.Stitcher_PANORAMA)
    
    try:
        # Stitch the images
        status, panorama = stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            print('Panorama stitching successful')
            return panorama
        else:
            print(f'Panorama stitching failed with status code: {status}')
            return None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    # Specify the folder containing your images
    image_folder = "old_images"  # Change this to your images folder path
    
    # Supported image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', 'JPG']
    
    # Get all image files from the folder
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(list(Path(image_folder).glob(f'*{ext}')))
    
    # Sort the images to ensure consistent ordering
    image_paths.sort()
    
    if not image_paths:
        print("No images found in the specified folder")
        return
    
    # Stitch the images
    panorama = stitch_images(image_paths)
    
    if panorama is not None:
        # Save the panorama
        output_path = "panorama_output.jpg"
        cv2.imwrite(output_path, panorama)
        print(f"Panorama saved as {output_path}")
        
        # Display the panorama (optional)
        # Resize for display if the image is too large
        height, width = panorama.shape[:2]
        max_display_width = 1200
        if width > max_display_width:
            scale = max_display_width / width
            display_size = (int(width * scale), int(height * scale))
            display_image = cv2.resize(panorama, display_size)
        else:
            display_image = panorama
            
        cv2.imshow('Panorama', display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()