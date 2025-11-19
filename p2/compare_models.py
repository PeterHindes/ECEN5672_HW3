import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading models...")
print("  - Model A (clean-trained, NO noise)...")
model_a = keras.models.load_model("mnist_autoencoder_clean_model.keras")
print("  - Model C (denoiser, trained WITH noise)...")
model_c = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Models loaded successfully!\n")

# Load test data
print("Loading test data...")
ds_test = tfds.load(
    "mnist", split="test[:200]", shuffle_files=True, as_supervised=False
)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    image = tf.reshape(image, (28, 28, 1))
    return image


test_images = []
for data in ds_test:
    test_images.append(normalize_img(data))
test_images = tf.stack(test_images)
print(f"✓ Loaded {len(test_images)} test images\n")

# Build models
print("Building models...")
dummy_input = tf.expand_dims(test_images[0], 0)
_ = model_a(dummy_input)
_ = model_c(dummy_input)
print("✓ Models built\n")

# Test noise levels (standard deviation)
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print(f"Testing {len(noise_levels)} noise levels: {noise_levels}")
print(f"Using {len(test_images)} test images\n")

results_a = {noise: [] for noise in noise_levels}
results_c = {noise: [] for noise in noise_levels}

print("Running noise comparison...")
for i, image in enumerate(test_images):
    if (i + 1) % 50 == 0:
        print(f"  Processed {i + 1}/{len(test_images)} images...")

    for noise_std in noise_levels:
        # Add noise to image
        if noise_std > 0:
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_std)
            noisy_image = image + noise
            noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
        else:
            noisy_image = image

        noisy_batch = tf.expand_dims(noisy_image, 0)

        # Model A
        recon_a = model_a.predict(noisy_batch, verbose=0)
        mse_a = np.mean((image.numpy() - recon_a[0]) ** 2)
        results_a[noise_std].append(mse_a)

        # Model C
        recon_c = model_c.predict(noisy_batch, verbose=0)
        mse_c = np.mean((image.numpy() - recon_c[0]) ** 2)
        results_c[noise_std].append(mse_c)

print("✓ Noise comparison complete\n")

# Calculate average MSE for each noise level
avg_mse_a = [np.mean(results_a[noise]) for noise in noise_levels]
avg_mse_c = [np.mean(results_c[noise]) for noise in noise_levels]

# Calculate improvement (positive = C is better)
improvements = []
for i in range(len(noise_levels)):
    if avg_mse_a[i] == 0:
        improvements.append(0.0)
    else:
        improvements.append((avg_mse_a[i] - avg_mse_c[i]) / avg_mse_a[i] * 100)

# Print summary
print("=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Noise σ':<10} {'Model A MSE':<15} {'Model C MSE':<15} {'Improvement':<15}")
print("-" * 70)
for i, noise in enumerate(noise_levels):
    imp_str = f"{improvements[i]:+.1f}%" if improvements[i] != 0 else "N/A"
    print(f"{noise:<10.1f} {avg_mse_a[i]:<15.6f} {avg_mse_c[i]:<15.6f} {imp_str:>15}")
print("=" * 70)

# Get example images for visualization
example_idx = np.random.randint(0, len(test_images))
example_image = test_images[example_idx]

# Get reconstructions at different noise levels
example_noises = [0.0, 0.3, 0.5]  # Show 3 examples
example_recons_a = []
example_recons_c = []
example_mses_a = []
example_mses_c = []
example_noisy = []

for noise_std in example_noises:
    if noise_std > 0:
        noise = tf.random.normal(
            shape=tf.shape(example_image), mean=0.0, stddev=noise_std
        )
        noisy_image = example_image + noise
        noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)
    else:
        noisy_image = example_image

    example_noisy.append(noisy_image)

    noisy_batch = tf.expand_dims(noisy_image, 0)

    recon_a = model_a.predict(noisy_batch, verbose=0)
    recon_c = model_c.predict(noisy_batch, verbose=0)

    example_recons_a.append(recon_a[0])
    example_recons_c.append(recon_c[0])
    example_mses_a.append(np.mean((example_image.numpy() - recon_a[0]) ** 2))
    example_mses_c.append(np.mean((example_image.numpy() - recon_c[0]) ** 2))

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 16))
gs = fig.add_gridspec(5, 1, height_ratios=[1.2, 1.2, 1.2, 1.2, 1.2], hspace=0.8)

# Plot 1: Performance comparison line plot
ax1 = fig.add_subplot(gs[0])
ax1.plot(
    noise_levels,
    avg_mse_a,
    "o-",
    linewidth=2.5,
    markersize=8,
    label="Model A (Clean-trained)",
    color="#e74c3c",
)
ax1.plot(
    noise_levels,
    avg_mse_c,
    "s-",
    linewidth=2.5,
    markersize=8,
    label="Model C (Denoiser)",
    color="#27ae60",
)
ax1.set_xlabel("Noise Level (stddev)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Average MSE", fontsize=12, fontweight="bold")
ax1.set_title(
    "Reconstruction Performance vs Noise Level", fontsize=14, fontweight="bold"
)
ax1.legend(fontsize=11, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_xticks(noise_levels)

# Plot 2: Improvement bar chart
ax2 = fig.add_subplot(gs[1])
colors = ["#c0392b" if x < 0 else "#27ae60" for x in improvements]
bars = ax2.bar(
    range(len(noise_levels)),
    improvements,
    color=colors,
    alpha=0.7,
    edgecolor="black",
)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax2.set_xlabel("Noise Level (stddev)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
ax2.set_title(
    "Performance Improvement of Denoiser over Clean-trained Model",
    fontsize=14,
    fontweight="bold",
)
ax2.set_xticks(range(len(noise_levels)))
ax2.set_xticklabels([f"σ={noise:.1f}" for noise in noise_levels])
ax2.grid(axis="y", alpha=0.3)
ax2.margins(y=0.2)

# Add percentage labels on bars
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    y_offset = 1.0 if height > 0 else -1.0
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + y_offset,
        f"{imp:+.1f}%",
        ha="center",
        va="bottom" if height > 0 else "top",
        fontsize=9,
        fontweight="bold",
    )

# Plot 3: Noisy inputs
ax3 = fig.add_subplot(gs[2])
ax3.axis("off")
ax3.set_title(f"Noisy Input Images", fontsize=13, fontweight="bold", pad=30)

gs3 = gs[2].subgridspec(1, len(example_noises) + 1, wspace=0.3)
ax_orig = fig.add_subplot(gs3[0])
ax_orig.imshow(example_image.numpy().squeeze(), cmap="gray")
ax_orig.set_title("Clean", fontsize=11, fontweight="bold")
ax_orig.axis("off")

for i, (noise, noisy) in enumerate(zip(example_noises, example_noisy)):
    ax = fig.add_subplot(gs3[i + 1])
    ax.imshow(noisy.numpy().squeeze(), cmap="gray")
    ax.set_title(f"Noise σ={noise:.1f}", fontsize=11, fontweight="bold")
    ax.axis("off")

# Plot 4: Model A reconstructions
ax4 = fig.add_subplot(gs[3])
ax4.axis("off")
ax4.set_title(
    f"Model A (Clean-trained) Reconstructions", fontsize=13, fontweight="bold", pad=30
)

gs4 = gs[3].subgridspec(1, len(example_noises) + 1, wspace=0.3)
ax_orig2 = fig.add_subplot(gs4[0])
ax_orig2.imshow(example_image.numpy().squeeze(), cmap="gray")
ax_orig2.set_title("Original", fontsize=11, fontweight="bold")
ax_orig2.axis("off")

for i, (noise, recon, mse) in enumerate(
    zip(example_noises, example_recons_a, example_mses_a)
):
    ax = fig.add_subplot(gs4[i + 1])
    ax.imshow(recon.squeeze(), cmap="gray")
    ax.set_title(f"σ={noise:.1f}\nMSE: {mse:.4f}", fontsize=11, fontweight="bold")
    ax.axis("off")

# Plot 5: Model C reconstructions
ax5 = fig.add_subplot(gs[4])
ax5.axis("off")
ax5.set_title(
    f"Model C (Denoiser) Reconstructions", fontsize=13, fontweight="bold", pad=30
)

gs5 = gs[4].subgridspec(1, len(example_noises) + 1, wspace=0.3)
ax_orig3 = fig.add_subplot(gs5[0])
ax_orig3.imshow(example_image.numpy().squeeze(), cmap="gray")
ax_orig3.set_title("Original", fontsize=11, fontweight="bold")
ax_orig3.axis("off")

for i, (noise, recon, mse) in enumerate(
    zip(example_noises, example_recons_c, example_mses_c)
):
    ax = fig.add_subplot(gs5[i + 1])
    ax.imshow(recon.squeeze(), cmap="gray")
    ax.set_title(f"σ={noise:.1f}\nMSE: {mse:.4f}", fontsize=11, fontweight="bold")
    ax.axis("off")

plt.suptitle(
    f"Autoencoder Comparison: Clean-trained vs Denoiser\n"
    f"Tested on {len(test_images)} images across {len(noise_levels)} noise levels",
    fontsize=15,
    fontweight="bold",
    y=0.995,
)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("noise_comparison.png", dpi=150, bbox_inches="tight")
print("\n✓ Saved visualization to: noise_comparison.png")
plt.show()

print("\n✓ Comparison complete!")
