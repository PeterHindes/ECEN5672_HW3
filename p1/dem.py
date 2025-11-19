import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt

def nlm_filter_numpy(image, h=10, patch_size=7, search_size=21):
    """
    Args:
        image: 2D numpy array (grayscale), normalized 0-255 or 0-1
        h: Decay parameter (larger = smoother but blurrier)
        patch_size: Size of the patches to compare (odd number)
        search_size: Max distance to search for similar patches (odd number)
    """
    # 1. Pre-processing
    # Convert to float for precision
    img = image.astype(np.float64)
    H, W = img.shape
    
    # Pad the image to handle boundaries for the search window
    # We pad with reflection to avoid artifacts at edges
    pad_r = search_size // 2
    img_padded = np.pad(img, pad_r, mode='reflect')
    
    # Output accumulators
    # 'Z' is the normalizing constant (sum of weights)
    denoised_img = np.zeros_like(img)
    Z = np.zeros_like(img)
    
    # The core image view (center) to compare against
    # This effectively slices out the original image from the center of the padding
    source_view = img_padded[pad_r:pad_r+H, pad_r:pad_r+W]

    # 2. Iterate over the Search Window (Offset Vectors)
    # We are shifting the "neighbor" view relative to the source
    # If search_size is 21, we loop -10 to +10
    search_radius = search_size // 2
    
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            
            # Optimization: Skip the center pixel (distance is 0, weight is 1)
            # handled implicitly, but explicit skip is sometimes faster.
            # Here we process it to ensure self-similarity is included.
            
            # Extract the shifted view (neighbor)
            # If dy=-1, dx=0, we are looking at the pixel above
            neighbor_view = img_padded[
                pad_r + dy : pad_r + dy + H,
                pad_r + dx : pad_r + dx + W
            ]
            
            # 3. Compute Patch Distances (Vectorized)
            # Squared difference between the pixel and its neighbor
            diff_sq = (source_view - neighbor_view) ** 2
            
            # Apply box filter to sum differences over the patch area.
            # uniform_filter computes the mean, but since the weights are relative,
            # mean vs sum doesn't change the relative decay logic, just the scaling of 'h'.
            patch_dist = uniform_filter(diff_sq, size=patch_size)
            
            # 4. Compute Weights
            # Formula: w(p,q) = exp( - ||Patch_p - Patch_q||^2 / h^2 )
            # We normalize by patch_size because uniform_filter calculated the mean, not sum
            w = np.exp(-np.maximum(patch_dist - 2 * (np.mean(img)**2), 0.0) / (h * h))
            
            # 5. Accumulate
            denoised_img += neighbor_view * w
            Z += w

    # 6. Normalize
    # Avoid division by zero
    return denoised_img / (Z + 1e-8)

# --- Example Usage ---
# Create a noisy image
original = np.zeros((100, 100))
original[20:60, 20:60] = 100
noise = np.random.normal(0, 10, original.shape)
noisy_image = original + noise

# Apply Filter
output = nlm_filter_numpy(noisy_image, h=10, patch_size=5, search_size=11)

# Display Results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.show()

# apply again
output = nlm_filter_numpy(output, h=10, patch_size=5, search_size=11)

# Display Results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(output, cmap='gray')
plt.title('Denoised Image')
plt.axis('off')

plt.show()

print("bye")