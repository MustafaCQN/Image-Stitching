from PIL import Image

# Open the original image
image = Image.open("panorama_output.jpg")  # Replace with your image file path

# Get image dimensions
width, height = image.size

# Create a new image for the result
output_image = Image.new("L", (width, height))  # 'L' mode is for grayscale (black and white)

# Process the image pixel by pixel
for y in range(height):
    for x in range(width):
        r, g, b = image.getpixel((x, y))[:3]  # Get RGB values (ignoring alpha if it exists)
        
        # Check if the pixel is black (0, 0, 0)
        if r == 0 and g == 0 and b == 0:
            output_image.putpixel((x, y), 0)  # White pixel for black regions in original
        else:
            output_image.putpixel((x, y), 255)  # Black pixel for non-black regions

# Save the new image
output_image.save("panorama_mask.jpg")  # Replace with your desired output file path

print("Black-and-white image created successfully!")
