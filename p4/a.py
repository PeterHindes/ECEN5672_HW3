import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.ops.math_ops import Mean

# Load the image
img = Image.open('img/Squirrel.jpeg')
# Convert to grayscale
img = img.convert('L')
# Convert to numpy array
img_array = np.array(img)

# Normalize the image
img_array = img_array / 255.0

# Batch into non overlapping 4x4 tiles
img_array_tiles = img_array.reshape(-1, 4, 4)


# Display the first tile to test
# plt.imshow(img_array[0], cmap='gray')
# plt.show()

tile_stats = []

# For each tile
for tile in img_array_tiles:
    # find the mean and standard deviation
    mean = np.mean(tile)
    std = np.std(tile)
    
    # map each pixel by checking if it is greater than the mean
    tile[tile >= mean] = 1
    tile[tile < mean] = 0
    
    # Append tile statistics to list
    tile_stats.append((mean, std))
    
# Display the first tile to test
# plt.imshow(img_array[0], cmap='gray')
# plt.show()

mean_bits = 5.0
std_bits = 3.0

img_array_tiles_quantized = np.copy(img_array_tiles)
for i, tile in enumerate(img_array_tiles):
    mean = tile_stats[i][0]
    std = tile_stats[i][1]
    tile[tile == 1] = mean + 0.674 * std
    tile[tile == 0] = mean - 0.674 * std
for i, tile_quantized in enumerate(img_array_tiles_quantized):
    mean = tile_stats[i][0]
    std = tile_stats[i][1]
    mean_quantized = int(mean*2.0**mean_bits)/(2.0**mean_bits)
    std_quantized = int(std*2.0**std_bits)/(2.0**std_bits)
    tile_quantized[tile_quantized == 1] = mean_quantized + 0.674 * std_quantized
    tile_quantized[tile_quantized == 0] = mean_quantized - 0.674 * std_quantized
    img_array_tiles_quantized[i] = tile_quantized
    
# Show the entire image and grayscale and difference all together
# plt.imshow(img_array_tiles.reshape(img_array.shape[0], img_array.shape[1]), cmap='gray')
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(np.array(img), cmap='gray')
ax[0].set_title('Grayscale Image')

# Calculate compression ratio
total_pixels = img_array.size
original_bits = total_pixels * 8
num_tiles = len(img_array_tiles)
bits_per_tile = mean_bits + std_bits + 16
compressed_bits = num_tiles * bits_per_tile
compression_ratio = original_bits / compressed_bits
bits_per_tile_unquant = 64 + 64 + 16
compressed_bits_unquant = num_tiles * bits_per_tile_unquant
compression_ratio_unquant = original_bits / compressed_bits_unquant

ax[1].imshow(img_array_tiles.reshape(img_array.shape[0], img_array.shape[1]), cmap='gray')
ax[1].set_title(f'BTC Image (CR: {compression_ratio_unquant:.2f})')

ax[2].imshow(img_array_tiles_quantized.reshape(img_array.shape[0], img_array.shape[1]), cmap='gray')
ax[2].set_title(f'BTC Quantized Image (CR: {compression_ratio:.2f})')

ax[3].imshow(img_array_tiles.reshape(img_array.shape[0], img_array.shape[1]) - np.array(img), cmap='gray')
ax[3].set_title('Difference of BTC VS Original Images')

plt.show()

