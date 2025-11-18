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

# NL Means Filter
def nlmean_filter(image, patch_size, window_size):
    if patch_size % 2 == 0:
        raise ValueError("patch_size must be odd")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    
    # Pad the image
    patch_width = ((patch_size-1) // 2)
    window_width = ((window_size-1) // 2)
    pad_width = patch_width + window_width
    padded_image = np.pad(image, pad_width=pad_width, mode='reflect')
    
    # For each pixel search the search window for best matches
    for x in range(pad_width,image.shape[0]+pad_width):
        for y in range(pad_width,image.shape[1]+pad_width):
            target_patch = padded_image[x-patch_width:x+patch_width, y-patch_width:y+patch_width]
            
            # For each pixel in the window compare the patch arround it to our primary patch
            for i in range(x-window_width, x+window_width+1):
                for j in range(y-window_width, y+window_width+1):
                    test_patch = padded_image[i-patch_width:i+patch_width, j-patch_width:j+patch_width]
                    
                    distances = np.linalg.norm(target_patch - test_patch)
                    
    
    return image

# apply filter
filtered = nlmean_filter(img_array, patch_size=3, window_size=11)#, mean_size=15)

# Display
plt.imshow(filtered, cmap='gray')
plt.show()

# Save
plt.imsave('filtered_image.png', filtered, cmap='gray')
