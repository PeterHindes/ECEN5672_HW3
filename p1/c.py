from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Load an image
img = Image.open('./output_noisy_image.jpg')

# Convert PIL image to numpy array for matplotlib
img_array = np.array(img)

def nlmean_filter(image, h=10, patch_size=7, search_size=21):
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

# Initial parameters
init_patch_size = 3
init_window_size = 11

# apply filter initially
filtered = nlmean_filter(img_array, h=10, patch_size=init_patch_size, search_size=init_window_size)

# Load Ground Truth
try:
    ground_truth = np.array(Image.open('./output_image.jpg'))
except FileNotFoundError:
    ground_truth = None
    print("output_image.jpg not found.")

# Calculate Difference
difference = None
if ground_truth is not None and ground_truth.shape == filtered.shape:
    difference = np.abs(ground_truth - filtered)

# Visualization
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
plt.subplots_adjust(bottom=0.25) # Make room for sliders

# Define zoom region (center 100x100)
h_img, w_img = img_array.shape
zoom_size = 100
zoom_r_start = h_img // 2 - zoom_size // 2
zoom_r_end = zoom_r_start + zoom_size
zoom_c_start = w_img // 2 - zoom_size // 2
zoom_c_end = zoom_c_start + zoom_size
zoom_slice = (slice(zoom_r_start, zoom_r_end), slice(zoom_c_start, zoom_c_end))

# Store image artists to update them later
artists = {}

# Store rectangles to update them
rects = []

# Helper to plot
def plot_pair(ax_full, ax_zoom, image, title, key, cmap='gray', colorbar=False):
    # Full
    if image is not None:
        im = ax_full.imshow(image, cmap=cmap)
        ax_full.set_title(title)
        if colorbar:
            plt.colorbar(im, ax=ax_full, fraction=0.046, pad=0.04)
        
        # Zoom
        im_zoom = ax_zoom.imshow(image[zoom_slice], cmap=cmap)
        ax_zoom.set_title(f'{title} (Zoom)')
        
        # Draw rectangle on full image to show zoom area
        rect = plt.Rectangle((zoom_c_start, zoom_r_start), zoom_size, zoom_size, 
                             linewidth=2, edgecolor='r', facecolor='none')
        ax_full.add_patch(rect)
        rects.append(rect)
        
        artists[key] = {'full': im, 'zoom': im_zoom, 'image': image}
    else:
        ax_full.text(0.5, 0.5, 'N/A', ha='center')
        ax_full.set_title(title)
        ax_zoom.text(0.5, 0.5, 'N/A', ha='center')
        ax_zoom.set_title(f'{title} (Zoom)')
    
    ax_full.axis('off')
    ax_zoom.axis('off')

# 1. Noisy Image
plot_pair(axes[0, 0], axes[1, 0], img_array, 'Noisy Image', 'noisy')

# 2. Ground Truth
plot_pair(axes[0, 1], axes[1, 1], ground_truth, 'Ground Truth', 'gt')

# 3. Filtered Image
plot_pair(axes[0, 2], axes[1, 2], filtered, 'Best Match Filtered', 'filtered')

# 4. Difference Image
plot_pair(axes[0, 3], axes[1, 3], difference, 'Difference', 'diff', cmap='hot', colorbar=True)

# Interactive Zoom
is_dragging = False

def update_zoom_view():
    global zoom_slice
    zoom_slice = (slice(zoom_r_start, zoom_r_end), slice(zoom_c_start, zoom_c_end))
    
    # Update all zoom images
    for key, art in artists.items():
        if 'image' in art and art['image'] is not None:
            art['zoom'].set_data(art['image'][zoom_slice])
            
    # Update all rectangles
    for rect in rects:
        rect.set_xy((zoom_c_start, zoom_r_start))
        
    fig.canvas.draw_idle()

def update_zoom_from_event(event):
    global zoom_r_start, zoom_r_end, zoom_c_start, zoom_c_end
    if event.xdata is None or event.ydata is None:
        return
        
    # Center the zoom on the mouse click
    c_center = int(event.xdata)
    r_center = int(event.ydata)
    
    # Clamp to image bounds
    zoom_r_start = max(0, min(h_img - zoom_size, r_center - zoom_size // 2))
    zoom_c_start = max(0, min(w_img - zoom_size, c_center - zoom_size // 2))
    zoom_r_end = zoom_r_start + zoom_size
    zoom_c_end = zoom_c_start + zoom_size
    
    update_zoom_view()

def on_press(event):
    global is_dragging
    # Check if click is in any of the top row axes
    if event.inaxes in axes[0, :]:
        is_dragging = True
        update_zoom_from_event(event)

def on_motion(event):
    if is_dragging and event.inaxes in axes[0, :]:
        update_zoom_from_event(event)

def on_release(event):
    global is_dragging
    is_dragging = False

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# Sliders
ax_patch = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_window = plt.axes([0.25, 0.05, 0.65, 0.03])

s_patch = Slider(ax_patch, 'Patch Size', 3, 7, valinit=init_patch_size, valstep=2)
s_window = Slider(ax_window, 'Window Size', 11, 21, valinit=init_window_size, valstep=2)

def update(val):
    p_size = int(s_patch.val)
    w_size = int(s_window.val)
    
    # Re-run filter
    print(f"Updating... Patch: {p_size}, Window: {w_size}")
    new_filtered = nlmean_filter(img_array, h=10, patch_size=p_size, search_size=w_size)
    
    # Update Filtered images
    artists['filtered']['full'].set_data(new_filtered)
    artists['filtered']['zoom'].set_data(new_filtered[zoom_slice])
    
    # Update Difference images
    if ground_truth is not None:
        new_diff = np.abs(ground_truth - new_filtered)
        artists['diff']['full'].set_data(new_diff)
        artists['diff']['zoom'].set_data(new_diff[zoom_slice])
        
        # Update color limits for difference map if needed (optional, but good for contrast)
        artists['diff']['full'].set_clim(vmin=new_diff.min(), vmax=new_diff.max())
        artists['diff']['zoom'].set_clim(vmin=new_diff.min(), vmax=new_diff.max())

    fig.canvas.draw_idle()
    
    # Save results on update (optional, but requested "save the nl filtered image")
    plt.imsave('filtered_image.png', new_filtered, cmap='gray')
    if ground_truth is not None:
        plt.imsave('difference_image.png', np.abs(ground_truth - new_filtered), cmap='hot')

s_patch.on_changed(update)
s_window.on_changed(update)

plt.savefig('comparison_plot.png')
plt.show()

print("Saved filtered_image.png, difference_image.png, and comparison_plot.png")
