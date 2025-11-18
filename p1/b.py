import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

# Load an image
img = Image.open('./output_noisy_image.jpg')

# Convert PIL image to numpy array for matplotlib
img_array = np.array(img)

# Display the grayscale image
plt.figure(figsize=(10, 8))
plt.imshow(img_array, cmap='gray')
plt.axis('off')
plt.title('Noisy Image')
plt.show()

# Convolution operations
# Mean Filter

def mean_filter(image, kernel_size):
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    return convolve2d(image, kernel, mode='same', boundary='symm')

# Gaussian Filter

def gaussian_filter(image, kernel_size):
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Calculate sigma from the kernel size so that the bell curve optimally covers the window size
    sigma = kernel_size / 6.0
    
    # Create a 2D Gaussian kernel
    x, y = np.mgrid[-kernel_size//2 + 1:kernel_size//2 + 1, -kernel_size//2 + 1:kernel_size//2 + 1]
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    g /= g.sum()
    
    return convolve2d(image, g, mode='same', boundary='symm')

# Apply the filters
windowsize = 15
mean_filtered = mean_filter(img_array, windowsize)
gaussian_filtered = gaussian_filter(img_array, windowsize)

# Display the filtered images
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(mean_filtered, cmap='gray')
plt.axis('off')
plt.title('Mean Filtered Image')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_filtered, cmap='gray')
plt.axis('off')
plt.title('Gaussian Filtered Image')

plt.show()

# Save the filtered images
plt.imsave('mean_filtered.jpg', mean_filtered, cmap='gray')
plt.imsave('gaussian_filtered.jpg', gaussian_filtered, cmap='gray')
