import cv2
import numpy as np
import os

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def stitch_images(images):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Start with the first image as the base
    base_image = images[0]
    
    for i in range(1, len(images)):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)

        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for j, match in enumerate(matches):
            points1[j, :] = keypoints1[match.queryIdx].pt
            points2[j, :] = keypoints2[match.trainIdx].pt

        # Find homography matrix
        homography_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Warp the base image to align with the new image
        height1, width1 = base_image.shape[:2]
        height2, width2 = images[i].shape[:2]

        # Calculate corners of the panorama
        corners1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]], dtype=np.float32)
        corners1 = corners1.reshape(-1, 1, 2)
        transformed_corners1 = cv2.perspectiveTransform(corners1, homography_matrix)

        # Get the dimensions of the new panorama
        all_corners = np.vstack((
            transformed_corners1.reshape(-1, 2),
            [[0, 0], [0, height2], [width2, height2], [width2, 0]]
        ))
        [x_min, y_min] = np.int32(all_corners.min(axis=0))
        [x_max, y_max] = np.int32(all_corners.max(axis=0))

        # Adjust translation
        translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        panorama_width = x_max - x_min
        panorama_height = y_max - y_min

        # Warp the base image to the new panorama
        warped_base = cv2.warpPerspective(base_image, translation_matrix @ homography_matrix, (panorama_width, panorama_height))

        # Place the new image on the panorama canvas
        panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
        panorama[-y_min:height2 - y_min, -x_min:width2 - x_min] = images[i]

        # Combine the base image and the new image
        base_image = np.where(warped_base > 0, warped_base, panorama)

    return base_image

if __name__ == "__main__":
    # Folder containing images
    folder_path = 'images'  # Replace with the folder containing your images
    images = load_images_from_folder(folder_path)

    if len(images) < 2:
        print("Need at least two images to stitch!")
    else:
        # Stitch the images together
        panorama = stitch_images(images)

        # Display the resulting panorama
        cv2.imshow('Panorama', panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally, save the panorama to a file
        cv2.imwrite('panorama_output.jpg', panorama)
