import cv2
import cv2.xphoto

original = cv2.imread('panorama_output.jpg')
mask = cv2.imread('new_panorama_mask.jpg', 0)

# Inpaint the image
inpaint_TELEA = cv2.inpaint(original, mask, 3, cv2.INPAINT_TELEA)
inpaint_NS = cv2.inpaint(original, mask, 3, cv2.INPAINT_NS)

# Display the results
# cv2.imshow('Original', original)
# cv2.imshow('Mask', mask)
cv2.imshow('Inpaint Telea', inpaint_TELEA)
# cv2.imshow('Inpaint NS', inpaint_NS)
cv2.waitKey(0)
cv2.destroyAllWindows()

