import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Loading saved denoising model...")
model = keras.models.load_model("mnist_autoencoder_denoiser_model_with_labels.keras")
print("✓ Model loaded successfully!\n")

# Find the layer right before the bottleneck
target_layer_name = None
for i, layer in enumerate(model.layers):
    if "encoder_bottleneck_output" in layer.name:
        # Get the previous conv layer
        for j in range(i - 1, -1, -1):
            if isinstance(
                model.layers[j], (keras.layers.Conv2D, keras.layers.LeakyReLU)
            ):
                if isinstance(model.layers[j], keras.layers.Conv2D):
                    target_layer_name = model.layers[j].name
                    break
                # If it's LeakyReLU, continue to find the Conv2D before it
                continue
        break

if target_layer_name is None:
    # Fallback: find the last Conv2D in the encoder
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            target_layer_name = layer.name

print(f"Visualizing effective filters for layer: {target_layer_name}\n")

# Create a model that outputs the activations of the target layer
layer_output = model.get_layer(target_layer_name).output
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer_output)


def compute_loss(input_image, filter_index):
    """Compute the mean activation of a specific filter"""
    activation = feature_extractor(input_image)
    # We want to maximize the mean activation of the target filter
    return tf.reduce_mean(activation[:, :, :, filter_index])


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    """Perform one gradient ascent step"""
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)

    # Compute gradients
    grads = tape.gradient(loss, img)

    # Normalize gradients
    grads = tf.math.l2_normalize(grads)

    # Update image and return new value
    updated_img = img + learning_rate * grads

    return updated_img, loss


def generate_pattern(filter_index, size=(28, 28), iterations=100, learning_rate=1.0):
    """Generate input pattern that maximally activates a filter"""
    # Start with random noise
    img = tf.random.uniform((1, size[0], size[1], 1), minval=0.4, maxval=0.6)

    # Gradient ascent
    for i in range(iterations):
        img, loss = gradient_ascent_step(img, filter_index, learning_rate)

        # Optional: Add small amount of regularization to keep values reasonable
        img = tf.clip_by_value(img, 0.0, 1.0)

    return img.numpy()[0, :, :, 0]


# Get number of filters in the target layer
num_filters = feature_extractor.output.shape[-1]
print(f"Number of filters in {target_layer_name}: {num_filters}")
print(f"Generating effective filter patterns...\n")

# Limit to 64 filters for visualization
num_filters_to_show = min(num_filters, 64)
grid_size = int(np.ceil(np.sqrt(num_filters_to_show)))

# Generate patterns
patterns = []
for filter_idx in range(num_filters_to_show):
    print(
        f"Generating pattern for filter {filter_idx + 1}/{num_filters_to_show}...",
        end="\r",
    )
    pattern = generate_pattern(filter_idx, iterations=150, learning_rate=1.5)
    patterns.append(pattern)

print("\n\nVisualizing patterns...")

# Create visualization
fig = plt.figure(figsize=(20, 20))
fig.suptitle(
    f'Effective Filters and Activations for Layer: "{target_layer_name}"\n'
    f"(Gradient-based visualization showing input patterns that maximally activate each filter)",
    fontsize=16,
    fontweight="bold",
)

for i, pattern in enumerate(patterns):
    ax = plt.subplot(grid_size, grid_size, i + 1)

    # Normalize pattern for visualization
    pattern_normalized = (pattern - pattern.min()) / (
        pattern.max() - pattern.min() + 1e-8
    )

    ax.imshow(pattern_normalized, cmap="viridis", interpolation="bilinear")
    ax.set_title(f"Effective Filter {i}", fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("effective_filters_visualization.png", dpi=150, bbox_inches="tight")
print("✓ Saved visualization to: effective_filters_visualization.png")
plt.show()

print("\n" + "=" * 70)
print("Effective Filter Visualization Complete!")
print("=" * 70)
print("\nInterpretation Guide:")
print("  - Each image shows the INPUT PATTERN that maximally activates that filter")
print("  - These patterns reveal what features the filter is looking for")
print("  - Bright (yellow) areas: Where the filter expects high values")
print("  - Dark (purple) areas: Where the filter expects low values")
print("\nWhat to look for:")
print("  - Edge detectors: Show oriented lines/edges")
print("  - Blob detectors: Show circular or blob-like patterns")
print("  - Texture detectors: Show repetitive patterns")
print("  - Complex features: Show digit-like or curved patterns")
print("\nNote: This layer is right before the bottleneck, so these filters")
print("      represent the high-level features used for encoding/classification.")
