import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave

src = "img/Earth.jpg"
dst_dir = "p5"
os.makedirs(dst_dir, exist_ok=True)

img = imread(src)

# If image is already grayscale (2D), keep as-is. Otherwise convert RGB/RGBA to grayscale.
if img.ndim == 2:
    gray = img
else:
    # Drop alpha if present
    rgb = img[..., :3]

    # Normalize integer images to [0,1] for correct luminance calculation
    if np.issubdtype(rgb.dtype, np.integer):
        rgb = rgb.astype("float32") / 255.0
    else:
        rgb = rgb.astype("float32")

    # Standard luminance weights
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=rgb.dtype)
    gray = np.dot(rgb, weights)

# Prepare output array as uint8
if np.issubdtype(gray.dtype, np.floating):
    gray_out = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
else:
    gray_out = gray.astype(np.uint8)

out_path = os.path.join(dst_dir, "Earth_gray.jpg")
imsave(out_path, gray_out, cmap="gray")

# Partition the grayscale image into 8x8 blocks (padding the edges if necessary)
block_size = 8

# Ensure we have a 2D array
img_blocks_src = np.squeeze(gray_out)
if img_blocks_src.ndim == 3 and img_blocks_src.shape[2] == 1:
    img_blocks_src = img_blocks_src[..., 0]
elif img_blocks_src.ndim != 2:
    raise ValueError("Unexpected grayscale image shape: {}".format(img_blocks_src.shape))

h, w = img_blocks_src.shape

# Compute padding so dimensions are multiples of block_size
pad_h = (block_size - (h % block_size)) % block_size
pad_w = (block_size - (w % block_size)) % block_size

if pad_h or pad_w:
    img_padded = np.pad(img_blocks_src, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
else:
    img_padded = img_blocks_src

n_blocks_y = img_padded.shape[0] // block_size
n_blocks_x = img_padded.shape[1] // block_size

# Create directory for blocks
blocks_dir = os.path.join(dst_dir, "blocks_8x8")
os.makedirs(blocks_dir, exist_ok=True)

# Extract blocks and save each as an image; also store in a list/array
blocks = np.zeros((n_blocks_y, n_blocks_x, block_size, block_size), dtype=img_padded.dtype)
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        y0 = by * block_size
        x0 = bx * block_size
        block = img_padded[y0:y0 + block_size, x0:x0 + block_size].copy()
        blocks[by, bx] = block
        out_block_path = os.path.join(blocks_dir, f"block_{by:03d}_{bx:03d}.png")
        imsave(out_block_path, block, cmap="gray")

# blocks is a (n_blocks_y, n_blocks_x, 8, 8) array of uint8 blocks

# Toggle whether to create and display the mosaic of tiles
show_mosaic = False

if show_mosaic:
    # Show all tiles in a single tiled mosaic with minimal spacing between tiles
    sep = 1  # spacing in pixels between tiles (set to 0 for no spacing)
    sep_val = 255  # separator fill value (255 = white)

    # Compute mosaic size
    mosaic_h = n_blocks_y * block_size + max(0, n_blocks_y - 1) * sep
    mosaic_w = n_blocks_x * block_size + max(0, n_blocks_x - 1) * sep

    mosaic = np.full((mosaic_h, mosaic_w), fill_value=sep_val, dtype=blocks.dtype)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * (block_size + sep)
            x0 = bx * (block_size + sep)
            mosaic[y0:y0 + block_size, x0:x0 + block_size] = blocks[by, bx]

    print(f"Displaying mosaic of tiles: {n_blocks_y}x{n_blocks_x} (mosaic shape = {mosaic.shape}, dtype = {mosaic.dtype}, sep={sep})")

    # Choose a reasonable figure size (in inches), capped so it's not absurdly large
    fig_w = max(4, min(20, mosaic_w / 100))
    fig_h = max(4, min(20, mosaic_h / 100))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(mosaic, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
else:
    print(f"Skipping mosaic display (show_mosaic=False). Blocks grid: {n_blocks_y}x{n_blocks_x}, block_size={block_size}")


# Compute the 2D DCT (Type-II, orthonormal) of each block
# We'll build the DCT transform matrix of size block_size x block_size and apply:
#   DCT2(block) = D @ block @ D.T
N = block_size

# Create DCT transform matrix (orthonormal)
dct_mat = np.empty((N, N), dtype=np.float32)
n_idx = np.arange(N)
for k in range(N):
    if k == 0:
        alpha = np.sqrt(1.0 / N)
    else:
        alpha = np.sqrt(2.0 / N)
    dct_mat[k, :] = alpha * np.cos(np.pi * (2 * n_idx + 1) * k / (2.0 * N))

# Allocate array for DCT coefficients (float32)
dct_blocks = np.empty((n_blocks_y, n_blocks_x, N, N), dtype=np.float32)

# Compute DCT for each block
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        block = blocks[by, bx].astype(np.float32)
        # If you want JPEG-style centered DCT, uncomment the next line:
        # block = block - 128.0
        dct_block = dct_mat @ block @ dct_mat.T
        dct_blocks[by, bx] = dct_block

print(f"Computed 2D DCT for each block: dct_blocks.shape = {dct_blocks.shape}, dtype = {dct_blocks.dtype}")


show_dct_mosaic = False

if show_dct_mosaic:
    # Create a visual representation of the DCT coefficients.
    # We'll use log(1 + abs(coeff)) for visibility, and scale using robust percentiles.
    sep = 1  # spacing between tiles in pixels
    sep_val = 255  # separator fill (white)

    # Use magnitude and log scaling for display
    mag = np.abs(dct_blocks)  # shape = (n_blocks_y, n_blocks_x, N, N)
    mag_log = np.log1p(mag)

    # Robust min/max via percentiles to avoid outlier domination
    vmin = float(np.percentile(mag_log, 1.0))
    vmax = float(np.percentile(mag_log, 99.0))
    if vmax <= vmin:
        # fallback to actual min/max
        vmin = float(mag_log.min())
        vmax = float(mag_log.max())
    # Avoid division by zero
    if vmax == vmin:
        vmax = vmin + 1.0

    mosaic_h = n_blocks_y * N + max(0, n_blocks_y - 1) * sep
    mosaic_w = n_blocks_x * N + max(0, n_blocks_x - 1) * sep

    mosaic = np.full((mosaic_h, mosaic_w), fill_value=sep_val, dtype=np.uint8)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * (N + sep)
            x0 = bx * (N + sep)
            tile = mag_log[by, bx]
            # normalize to [0,255] using vmin/vmax
            tile_norm = np.clip((tile - vmin) / (vmax - vmin), 0.0, 1.0)
            tile_u8 = (tile_norm * 255.0).astype(np.uint8)
            mosaic[y0:y0 + N, x0:x0 + N] = tile_u8

    print(f"Displaying DCT mosaic: {n_blocks_y}x{n_blocks_x} tiles, tile_size={N}, mosaic_shape={mosaic.shape}, dtype={mosaic.dtype}")
    print(f"DCT log-scale vmin={vmin:.6g}, vmax={vmax:.6g} (percentiles 1/99)")

    # Choose a reasonable figure size (in inches), capped so it's not too large
    fig_w = max(4, min(20, mosaic_w / 100))
    fig_h = max(4, min(20, mosaic_h / 100))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(mosaic, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("2D DCT tiles (log-scaled magnitude)")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
else:
    print("Skipping DCT mosaic display (show_dct_mosaic=False).")


# Toggle whether to display DCT tiles one-by-one (interactive)
show_dct_one_by_one = False

if show_dct_one_by_one:
    print("Now displaying DCT tiles one-by-one.")
    print("Instructions: press Enter to advance to the next tile,")
    print("type a single space then press Enter to stop early, or press Ctrl+C to abort.")

    # Prepare log-magnitude scaling (robust percentiles as used in the mosaic code)
    mag = np.abs(dct_blocks)
    mag_log = np.log1p(mag)

    vmin = float(np.percentile(mag_log, 1.0))
    vmax = float(np.percentile(mag_log, 99.0))
    if vmax <= vmin:
        vmin = float(mag_log.min())
        vmax = float(mag_log.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    try:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                tile = mag_log[by, bx]
                tile_norm = np.clip((tile - vmin) / (vmax - vmin), 0.0, 1.0)
                tile_u8 = (tile_norm * 255.0).astype(np.uint8)

                fig = plt.figure(figsize=(4, 4))
                plt.imshow(tile_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                plt.title(f"2D DCT log-mag: block ({by},{bx}) — tile_size={N}")
                plt.axis("off")
                plt.tight_layout(pad=0)
                # non-blocking show so we can accept terminal input to continue/stop
                plt.show(block=False)

                try:
                    resp = input("Press Enter for next, or type a single space then Enter to stop (Ctrl+C to abort): ")
                except KeyboardInterrupt:
                    print("\nInterrupted by user (Ctrl+C). Stopping display.")
                    plt.close(fig)
                    raise

                plt.close(fig)

                if resp == " ":
                    print("Stop requested by user (space).")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        # Clean exit path when user requested stop
        print("Exiting per user request.")
else:
    print("Skipping one-by-one DCT display (show_dct_one_by_one=False).")
    print("Set show_dct_one_by_one = True to enable interactive display of DCT tiles.")


# Keep only the top-left 4x4 DCT coefficients in each block and reconstruct the image
keep_k = 4
print(f"Keeping top {keep_k}x{keep_k} DCT coefficients per block and zeroing the rest...")

# Mask out high-frequency coefficients
dct_masked = np.zeros_like(dct_blocks)
dct_masked[:, :, :keep_k, :keep_k] = dct_blocks[:, :, :keep_k, :keep_k]

# Inverse 2D DCT for each block: block = D.T @ coeff @ D
recon_blocks = np.empty_like(dct_blocks, dtype=np.float32)
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        coeff = dct_masked[by, bx]
        block_rec = dct_mat.T @ coeff @ dct_mat
        recon_blocks[by, bx] = block_rec

# Assemble the padded reconstructed image
img_rec_padded = np.zeros_like(img_padded, dtype=np.float32)
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        y0 = by * block_size
        x0 = bx * block_size
        img_rec_padded[y0:y0 + block_size, x0:x0 + block_size] = recon_blocks[by, bx]

# Crop to original image size
img_rec = img_rec_padded[:h, :w]

# Convert to uint8 image (round, clip)
img_rec_u8 = np.clip(np.rint(img_rec), 0, 255).astype(np.uint8)

# Save reconstructed image
out_recon_path = os.path.join(dst_dir, f"Earth_recon_{keep_k}x{keep_k}.jpg")
imsave(out_recon_path, img_rec_u8, cmap="gray")
print(f"Reconstructed image saved to: {out_recon_path}")

# Print a simple quality metric (MSE)
orig_f = img_blocks_src.astype(np.float32)
mse = float(np.mean((orig_f[:h, :w] - img_rec[:h, :w]) ** 2))
print(f"MSE between original grayscale and {keep_k}x{keep_k} reconstruction: {mse:.4f}")

# Display the reconstructed image
plt.figure(figsize=(8, 8 * (h / max(1, w))))
plt.imshow(img_rec_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
plt.title(f"Reconstruction using top {keep_k}x{keep_k} DCT coefficients")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# Now repeat the same pipeline but using the 2D discrete Fourier transform (DFT) instead of the DCT.
print("\n--- Re-running operations using 2D DFT (FFT-based) instead of DCT ---")

# We'll use numpy.fft.fft2 / ifft2 on each block.
N = block_size

# Allocate array for DFT coefficients (complex64)
dft_blocks = np.empty((n_blocks_y, n_blocks_x, N, N), dtype=np.complex64)

# Compute 2D DFT for each block
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        block = blocks[by, bx].astype(np.float32)
        # Compute forward 2D FFT (DC at index [0,0])
        coeff = np.fft.fft2(block)
        dft_blocks[by, bx] = coeff

print(f"Computed 2D DFT for each block: dft_blocks.shape = {dft_blocks.shape}, dtype = {dft_blocks.dtype}")

# Optionally display a mosaic of log-magnitude of DFT coefficients
show_dft_mosaic = False

if show_dft_mosaic:
    sep = 1
    sep_val = 255

    mag = np.abs(dft_blocks)
    mag_log = np.log1p(mag)

    vmin = float(np.percentile(mag_log, 1.0))
    vmax = float(np.percentile(mag_log, 99.0))
    if vmax <= vmin:
        vmin = float(mag_log.min())
        vmax = float(mag_log.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    mosaic_h = n_blocks_y * N + max(0, n_blocks_y - 1) * sep
    mosaic_w = n_blocks_x * N + max(0, n_blocks_x - 1) * sep

    mosaic = np.full((mosaic_h, mosaic_w), fill_value=sep_val, dtype=np.uint8)

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * (N + sep)
            x0 = bx * (N + sep)
            tile = mag_log[by, bx]
            tile_norm = np.clip((tile - vmin) / (vmax - vmin), 0.0, 1.0)
            tile_u8 = (tile_norm * 255.0).astype(np.uint8)
            mosaic[y0:y0 + N, x0:x0 + N] = tile_u8

    print(f"Displaying DFT mosaic: {n_blocks_y}x{n_blocks_x} tiles, tile_size={N}, mosaic_shape={mosaic.shape}")
    fig_w = max(4, min(20, mosaic_w / 100))
    fig_h = max(4, min(20, mosaic_h / 100))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(mosaic, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    plt.title("2D DFT tiles (log-scaled magnitude)")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
else:
    print("Skipping DFT mosaic display (show_dft_mosaic=False).")

# One-by-one DFT tiles viewing (interactive)
show_dft_one_by_one = True

if show_dft_one_by_one:
    print("Now displaying DFT tiles one-by-one (log-magnitude).")
    mag = np.abs(dft_blocks)
    mag_log = np.log1p(mag)

    vmin = float(np.percentile(mag_log, 1.0))
    vmax = float(np.percentile(mag_log, 99.0))
    if vmax <= vmin:
        vmin = float(mag_log.min())
        vmax = float(mag_log.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    try:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                tile = mag_log[by, bx]
                tile_norm = np.clip((tile - vmin) / (vmax - vmin), 0.0, 1.0)
                tile_u8 = (tile_norm * 255.0).astype(np.uint8)

                fig = plt.figure(figsize=(4, 4))
                plt.imshow(tile_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                plt.title(f"2D DFT log-mag: block ({by},{bx}) — tile_size={N}")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.show(block=False)

                try:
                    resp = input("Press Enter for next, or type a single space then Enter to stop (Ctrl+C to abort): ")
                except KeyboardInterrupt:
                    print("\nInterrupted by user (Ctrl+C). Stopping display.")
                    plt.close(fig)
                    raise

                plt.close(fig)

                if resp == " ":
                    print("Stop requested by user (space).")
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Exiting per user request.")
else:
    print("Skipping one-by-one DFT display (show_dft_one_by_one=False).")

# Keep only the top-left keep_k x keep_k DFT coefficients (low frequencies near [0,0]) and reconstruct
keep_k = 4
print(f"Keeping top {keep_k}x{keep_k} DFT coefficients per block and zeroing the rest...")

dft_masked = np.zeros_like(dft_blocks)
dft_masked[:, :, :keep_k, :keep_k] = dft_blocks[:, :, :keep_k, :keep_k]

# Inverse 2D DFT for each block: use ifft2, take real part
recon_blocks = np.empty((n_blocks_y, n_blocks_x, N, N), dtype=np.float32)
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        coeff = dft_masked[by, bx]
        block_rec = np.fft.ifft2(coeff)
        # numerical imaginary residuals should be near zero; take the real part
        recon_blocks[by, bx] = np.real(block_rec).astype(np.float32)

# Assemble the padded reconstructed image
img_rec_padded = np.zeros_like(img_padded, dtype=np.float32)
for by in range(n_blocks_y):
    for bx in range(n_blocks_x):
        y0 = by * block_size
        x0 = bx * block_size
        img_rec_padded[y0:y0 + block_size, x0:x0 + block_size] = recon_blocks[by, bx]

# Crop to original image size
img_rec = img_rec_padded[:h, :w]

# Convert to uint8 image (round, clip)
img_rec_u8 = np.clip(np.rint(img_rec), 0, 255).astype(np.uint8)

# Save reconstructed image
out_recon_path = os.path.join(dst_dir, f"Earth_recon_dft_{keep_k}x{keep_k}.jpg")
imsave(out_recon_path, img_rec_u8, cmap="gray")
print(f"DFT-based reconstructed image saved to: {out_recon_path}")

# Print a simple quality metric (MSE)
orig_f = img_blocks_src.astype(np.float32)
mse = float(np.mean((orig_f[:h, :w] - img_rec[:h, :w]) ** 2))
print(f"MSE between original grayscale and DFT {keep_k}x{keep_k} reconstruction: {mse:.4f}")

# Display the reconstructed image
plt.figure(figsize=(8, 8 * (h / max(1, w))))
plt.imshow(img_rec_u8, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
plt.title(f"DFT reconstruction using top {keep_k}x{keep_k} coefficients")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()