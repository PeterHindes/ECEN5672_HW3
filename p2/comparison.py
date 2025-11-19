import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("=" * 70)
print("COMPARISON: Clean-trained vs Noise-trained Autoencoder Performance")
print("=" * 70)

# Load both models
print("\nLoading models...")
print("  1. Model A (trained on clean images)")
model_clean = keras.models.load_model("mnist_autoencoder_model.keras")
print("  2. Model C (trained on noisy images)")
model_denoiser = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Both models loaded\n")

# Load test data
print("Loading test data...")
ds_test = tfds.load("mnist", split="test[:1000]", as_supervised=False)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    label = data["label"]
    return image, label


ds_test = ds_test.map(normalize_img).batch(1)
print("✓ Test data loaded\n")

# Test parameters
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print(f"Testing with noise levels: {noise_levels}\n")

# Collect results
results_clean = {noise: [] for noise in noise_levels}
results_denoiser = {noise: [] for noise in noise_levels}

print("Running evaluation...")
print("=" * 70)

num_samples = 200
sample_count = 0

for clean_image, label in ds_test:
    if sample_count >= num_samples:
        break

    # Test at each noise level
    for noise_level in noise_levels:
        # Add noise
        if noise_level > 0:
            noise = tf.random.normal(
                shape=tf.shape(clean_image), mean=0.0, stddev=noise_level
            )
            noisy_image = clean_image + noise
        else:
            noisy_image = clean_image

        # Get reconstructions from both models
        recon_clean = model_clean.predict(noisy_image, verbose=0)
        recon_denoiser = model_denoiser.predict(noisy_image, verbose=0)

        # Calculate MSE for both
        mse_clean = np.mean((clean_image.numpy() - recon_clean) ** 2)
        mse_denoiser = np.mean((clean_image.numpy() - recon_denoiser) ** 2)

        results_clean[noise_level].append(mse_clean)
        results_denoiser[noise_level].append(mse_denoiser)

    sample_count += 1
    if sample_count % 50 == 0:
        print(f"  Processed {sample_count}/{num_samples} samples")

print(f"✓ Evaluation complete ({sample_count} samples per noise level)\n")

# Calculate statistics
print("=" * 70)
print("RESULTS: Average Reconstruction MSE")
print("=" * 70)
print(
    f"{'Noise Level':<15} {'Model A (Clean)':<20} {'Model C (Denoiser)':<20} {'Improvement':<15}"
)
print("-" * 70)

avg_mse_clean = []
avg_mse_denoiser = []
improvements = []

for noise in noise_levels:
    avg_clean = np.mean(results_clean[noise])
    avg_denoiser = np.mean(results_denoiser[noise])
    improvement = ((avg_clean - avg_denoiser) / avg_clean * 100) if avg_clean > 0 else 0

    avg_mse_clean.append(avg_clean)
    avg_mse_denoiser.append(avg_denoiser)
    improvements.append(improvement)

    print(
        f"{noise:<15.1f} {avg_clean:<20.6f} {avg_denoiser:<20.6f} {improvement:+.2f}%"
    )

print("=" * 70)

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: Performance comparison line plot
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(
    noise_levels,
    avg_mse_clean,
    marker="o",
    linewidth=2,
    markersize=8,
    label="Model A (Clean-trained)",
    color="red",
    alpha=0.7,
)
ax1.plot(
    noise_levels,
    avg_mse_denoiser,
    marker="s",
    linewidth=2,
    markersize=8,
    label="Model C (Denoiser)",
    color="green",
    alpha=0.7,
)
ax1.set_xlabel("Noise Level (stddev)", fontsize=12)
ax1.set_ylabel("Average MSE", fontsize=12)
ax1.set_title(
    "Reconstruction Performance vs Noise Level", fontsize=14, fontweight="bold"
)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.05, max(noise_levels) + 0.05])

# Plot 2: Improvement percentage
ax2 = fig.add_subplot(gs[1, :])
colors = ["green" if imp > 0 else "red" for imp in improvements]
bars = ax2.bar(
    [str(n) for n in noise_levels],
    improvements,
    color=colors,
    alpha=0.7,
    edgecolor="black",
)
ax2.axhline(y=0, color="black", linewidth=1)
ax2.set_xlabel("Noise Level (stddev)", fontsize=12)
ax2.set_ylabel("Improvement (%)", fontsize=12)
ax2.set_title(
    "Performance Improvement of Denoiser over Clean-trained Model",
    fontsize=14,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3, axis="y")

# Add percentage labels on bars
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{imp:+.1f}%",
        ha="center",
        va="bottom" if height > 0 else "top",
        fontsize=10,
        fontweight="bold",
    )

# Plot 3-8: Example reconstructions at different noise levels
example_indices = [0, 1, 2]  # Show 3 examples
example_noise_levels = [0.0, 0.3, 0.5]  # Show 3 noise levels

# Get a few example images
example_images = []
for i, (clean_image, label) in enumerate(ds_test):
    if i in example_indices:
        example_images.append(clean_image)
    if len(example_images) >= len(example_indices):
        break

# Create example grid
for idx, noise_level in enumerate(example_noise_levels):
    if idx >= len(example_images):
        break

    clean_img = example_images[idx]

    # Add noise
    if noise_level > 0:
        noise = tf.random.normal(
            shape=tf.shape(clean_img), mean=0.0, stddev=noise_level
        )
        noisy_img = clean_img + noise
    else:
        noisy_img = clean_img

    # Get reconstructions
    recon_clean = model_clean.predict(noisy_img, verbose=0)
    recon_denoiser = model_denoiser.predict(noisy_img, verbose=0)

    # Plot
    row = 2
    col = idx

    ax = fig.add_subplot(gs[row, col])

    # Create composite image: noisy | clean-recon | denoiser-recon
    composite = np.concatenate(
        [
            noisy_img[0].numpy().squeeze(),
            recon_clean[0].squeeze(),
            recon_denoiser[0].squeeze(),
        ],
        axis=1,
    )

    ax.imshow(composite, cmap="gray")
    ax.axis("off")
    mse_c = np.mean((clean_img.numpy() - recon_clean) ** 2)
    mse_d = np.mean((clean_img.numpy() - recon_denoiser) ** 2)
    ax.set_title(
        f"Noise σ={noise_level:.1f}\nNoisy | Model A | Model C\n"
        f"MSE: {mse_c:.4f} vs {mse_d:.4f}",
        fontsize=9,
    )

plt.suptitle(
    "Autoencoder Comparison: Clean-trained vs Noise-trained\n"
    f"Tested on {num_samples} images across {len(noise_levels)} noise levels",
    fontsize=16,
    fontweight="bold",
)

plt.savefig("comparison_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ Visualization saved to: comparison_analysis.png")
plt.show()

# Additional statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Number of test samples: {num_samples}")
print(f"Noise levels tested: {len(noise_levels)}")
print(
    f"\nBest improvement: {max(improvements):.2f}% at noise level {noise_levels[improvements.index(max(improvements))]}"
)
print(
    f"Worst result: {min(improvements):.2f}% at noise level {noise_levels[improvements.index(min(improvements))]}"
)

# Calculate average improvement across all noise levels
avg_improvement = np.mean(improvements[1:])  # Exclude noise=0
print(f"\nAverage improvement (noise > 0): {avg_improvement:.2f}%")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
if avg_improvement > 10:
    print("The denoising model shows SIGNIFICANT improvement on noisy images.")
elif avg_improvement > 0:
    print("The denoising model shows moderate improvement on noisy images.")
else:
    print("The denoising model shows limited or no improvement.")

print("\nKey Finding:")
print(f"  At noise level 0.3 (matching training noise), the denoiser achieves")
idx_03 = noise_levels.index(0.3)
print(
    f"  {improvements[idx_03]:.1f}% better reconstruction than the clean-trained model."
)
print("=" * 70)
