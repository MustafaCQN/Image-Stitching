import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

# Load the pre-trained Stable Diffusion model for inpainting
model_id = "runwayml/stable-diffusion-inpainting"  # You can use other models if you prefer
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)

# If you have a GPU, move the pipeline to GPU for faster processing
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def nearest_divisible(target, divisor):
    # Handle the case where divisor is zero to avoid division by zero error
    if divisor == 0:
        raise ValueError("Divisor cannot be zero")

    # Find the remainder when dividing target by divisor
    remainder = target % divisor

    # If remainder is 0, the number is already divisible by the divisor
    if remainder == 0:
        return target

    # Calculate the nearest divisible number by either rounding up or down
    if remainder <= divisor / 2:
        return target - remainder  # Round down to the nearest divisible number
    else:
        return target + (divisor - remainder)  # Round up to the nearest divisible number


def fill_blank(image_path, mask_path):
    """
    Fills the blank areas of an image based on a mask.
    
    Parameters:
    - image_path: Path to the input image
    - mask_path: Path to the binary mask (black=missing, white=preserved)
    """
    # Open the image and mask
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Mask should be in grayscale
    
    # Ensure the mask is in the correct format (binary mask: 0 for missing, 255 for present)
    #mask = mask.point(lambda p: p > 128 and 255 or 0)
    
    # Use the inpainting pipeline to fill the missing areas
    result = pipe(prompt="A detailed scene", image=image, mask_image=mask, width=nearest_divisible(image.size[0], 8), height=nearest_divisible(image.size[1], 8)).images[0]
    
    # Display and save the result
    result.show()
    result.save("inpainted_image.png")

# Example usage:
image_path = "panorama_output.jpg"  # Provide the image URL or local path
mask_path = "panorama_mask.jpg"  # Provide the mask URL or local path

# Inpaint and save the result
fill_blank(image_path, mask_path)
