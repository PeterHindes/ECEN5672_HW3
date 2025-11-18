import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load an image
img = Image.open('./img/TOM.jpg')

# Convert PIL image to numpy array for matplotlib
img_array = np.array(img)

# Display the image
plt.figure(figsize=(10, 8))
plt.imshow(img_array)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Example image processing operations
# Convert to grayscale
img_gray = img.convert('L')
img_gray_array = np.array(img_gray)

# Display grayscale image
plt.figure(figsize=(10, 8))
plt.imshow(img_gray_array, cmap='gray')
plt.axis('off')
plt.title('Grayscale Image')
plt.show()

# Add Gaussian noise
noise = np.random.normal(0, 10, img_gray_array.shape)
img_noisy = img_gray_array + noise
img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)

# Display noisy image
plt.figure(figsize=(10, 8))
plt.imshow(img_noisy, cmap='gray')
plt.axis('off')
plt.title('Noisy Image')
plt.show()

# Save processed image
img_gray.save('output_image.jpg')
# Save noisy image
img_noisy = Image.fromarray(img_noisy)
img_noisy.save('output_noisy_image.jpg')