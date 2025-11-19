import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading model...")
model = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
print("✓ Model loaded\n")

# Load test data
print("Loading test data...")
ds_test = tfds.load("mnist", split="test", as_supervised=False)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    label = data["label"]
    return image, label


ds_test = ds_test.map(normalize_img).batch(1)

# Get a test image to build the model
for test_image, test_label in ds_test.take(1):
    break
print(f"Test image shape: {test_image.shape}\n")

# BUILD THE MODEL by calling it once
print("Building model...")
_ = model(test_image)  # This builds the model and defines inputs/outputs
print("✓ Model built\n")

# Now show model structure
print("Model structure:")
bottleneck_idx = None
conv_before_bottleneck_idx = None

for i, layer in enumerate(model.layers):
    try:
        output_shape = layer.output_shape
    except:
        output_shape = "unknown"

    print(
        f"  {i}: {layer.name:30s} {layer.__class__.__name__:20s} {str(output_shape):20s}"
    )

    # Auto-detect bottleneck
    if "bottleneck" in layer.name.lower():
        bottleneck_idx = i
        print(f"       ↑ BOTTLENECK DETECTED!")
    elif (
        isinstance(layer, keras.layers.Dense)
        and bottleneck_idx is None
        and i > len(model.layers) // 3
    ):
        # First Dense layer in latter part is likely bottleneck
        bottleneck_idx = i
        print(f"       ↑ POTENTIAL BOTTLENECK (Dense layer)")

    # Find last Conv layer before bottleneck or Flatten
    if isinstance(layer, keras.layers.Conv2D):
        if bottleneck_idx is None or i < bottleneck_idx:
            conv_before_bottleneck_idx = i

print(f"\nUsing:")
print(f"  Conv layer index: {conv_before_bottleneck_idx}")
print(f"  Bottleneck index: {bottleneck_idx}\n")

if bottleneck_idx is None:
    print("ERROR: Could not auto-detect bottleneck!")
    exit(1)

# =============================================================================
# PART 1: EFFECTIVE FILTER VISUALIZATION (Gradient Ascent)
# =============================================================================
print("=" * 70)
print("PART 1: Visualizing Effective Filters (Gradient Ascent)")
print("=" * 70)

if conv_before_bottleneck_idx is not None:
    conv_layer = model.layers[conv_before_bottleneck_idx]

    # Create model to extract conv features (this builds the layer)
    feature_extractor = keras.Model(inputs=model.inputs, outputs=conv_layer.output)

    # Now we can get the shape by calling it once
    temp_output = feature_extractor(test_image)
    num_filters = temp_output.shape[-1]

    print(f"Conv layer: {conv_layer.name}")
    print(f"Number of filters: {num_filters}")

    # Gradient ascent parameters
    num_filters_to_viz = min(16, num_filters)
    img_size = 28
    iterations = 100
    step_size = 1.0

    print(f"Visualizing {num_filters_to_viz} filters using gradient ascent...")
    print(f"Iterations: {iterations}, Step size: {step_size}\n")

    effective_filters = []

    for filter_idx in range(num_filters_to_viz):
        # Start with random noise
        input_img = tf.Variable(
            tf.random.uniform((1, img_size, img_size, 1), minval=0.4, maxval=0.6)
        )

        for iteration in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(input_img)
                activation = feature_extractor(input_img)
                # Maximize the mean activation of this specific filter
                filter_activation = tf.reduce_mean(activation[:, :, :, filter_idx])

            # Compute gradient
            grads = tape.gradient(filter_activation, input_img)

            # Normalize gradients
            grads = tf.math.l2_normalize(grads)

            # Update input image
            input_img.assign_add(grads * step_size)

            # Clip to valid range
            input_img.assign(tf.clip_by_value(input_img, 0.0, 1.0))

        effective_filters.append(input_img.numpy()[0])
        if (filter_idx + 1) % 4 == 0:
            print(f"  Completed {filter_idx + 1}/{num_filters_to_viz} filters")

    print("✓ Effective filter visualization complete\n")
else:
    effective_filters = None
    print("No convolutional layer found before bottleneck\n")

# =============================================================================
# PART 2: PER-DIGIT BOTTLENECK ACTIVATION ANALYSIS
# =============================================================================
print("=" * 70)
print("PART 2: Analyzing Bottleneck Activations per Digit Class")
print("=" * 70)

# Create bottleneck extraction model
bottleneck_model = keras.Model(
    inputs=model.inputs, outputs=model.layers[bottleneck_idx].output
)

# Collect examples from each digit class
num_classes = 10
examples_per_class = 100
print(f"Collecting {examples_per_class} examples per digit class...")

class_examples = {i: [] for i in range(num_classes)}
class_bottlenecks = {i: [] for i in range(num_classes)}

# Collect examples
for image, label in ds_test:
    label_val = label.numpy()[0]
    if len(class_examples[label_val]) < examples_per_class:
        class_examples[label_val].append(image)

    # Check if we have enough examples for all classes
    if all(len(examples) >= examples_per_class for examples in class_examples.values()):
        break

# Compute bottleneck activations for each class
print("Computing bottleneck activations...")
for digit in range(num_classes):
    examples = class_examples[digit]
    for example in examples:
        bottleneck_activation = bottleneck_model.predict(example, verbose=0)
        class_bottlenecks[digit].append(bottleneck_activation[0])
    print(f"  Digit {digit}: {len(class_bottlenecks[digit])} examples processed")

# Compute average bottleneck activation per class
avg_bottlenecks = {}
std_bottlenecks = {}
for digit in range(num_classes):
    activations = np.array(class_bottlenecks[digit])
    avg_bottlenecks[digit] = np.mean(activations, axis=0)
    std_bottlenecks[digit] = np.std(activations, axis=0)
    print(
        f"  Digit {digit} - Mean activation: {np.mean(avg_bottlenecks[digit]):.4f}, Std: {np.mean(std_bottlenecks[digit]):.4f}"
    )

print("✓ Bottleneck analysis complete\n")

# =============================================================================
# PART 3: COMPREHENSIVE VISUALIZATION
# =============================================================================
print("=" * 70)
print("Creating comprehensive visualization...")
print("=" * 70)

# Get one example reconstruction
test_reconstruction = model.predict(test_image, verbose=0)
test_bottleneck = bottleneck_model.predict(test_image, verbose=0)

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(5, 8, hspace=0.5, wspace=0.3, height_ratios=[1, 1, 1.5, 0.8, 0.8])

# Row 1-2: Effective Filters (if available) - 8 per row, 2 rows = 16 filters
if effective_filters is not None:
    for i in range(min(16, len(effective_filters))):
        row = i // 8  # 0 for first 8 filters, 1 for next 8
        col = i % 8  # 0-7 for column position
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(effective_filters[i].squeeze(), cmap="viridis")
        ax.axis("off")
        ax.set_title(f"Filter {i}", fontsize=9)

# Row 3: Heatmap of average bottleneck activations per digit
ax_heatmap = fig.add_subplot(gs[2, :])
bottleneck_dim = len(avg_bottlenecks[0])
heatmap_data = np.array([avg_bottlenecks[i] for i in range(num_classes)])

im = ax_heatmap.imshow(heatmap_data, cmap="RdYlBu_r", aspect="auto")
ax_heatmap.set_xlabel("Bottleneck Neuron Index", fontsize=11)
ax_heatmap.set_ylabel("Digit Class", fontsize=11)
ax_heatmap.set_title(
    f"Average Bottleneck Activation per Digit Class ({bottleneck_dim} neurons)",
    fontsize=12,
    fontweight="bold",
)
ax_heatmap.set_yticks(range(num_classes))
ax_heatmap.set_yticklabels([f"Digit {i}" for i in range(num_classes)])
plt.colorbar(im, ax=ax_heatmap, label="Activation")

# Add grid
ax_heatmap.set_xticks(np.arange(bottleneck_dim) - 0.5, minor=True)
ax_heatmap.set_yticks(np.arange(num_classes) - 0.5, minor=True)
ax_heatmap.grid(which="minor", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

# Row 4: Top activating neurons per digit
ax_top_neurons = fig.add_subplot(gs[3, :4])
# For each digit, find top 3 most active neurons
top_neurons_per_digit = {}
for digit in range(num_classes):
    top_3_indices = np.argsort(avg_bottlenecks[digit])[-3:][::-1]
    top_neurons_per_digit[digit] = top_3_indices

# Create visualization
y_pos = np.arange(num_classes)
labels = []
for digit in range(num_classes):
    top_neurons = top_neurons_per_digit[digit]
    neuron_str = ", ".join([f"{n}" for n in top_neurons])
    labels.append(f"Digit {digit}: [{neuron_str}]")

ax_top_neurons.barh(y_pos, [1] * num_classes, alpha=0)  # Invisible bars for labels
ax_top_neurons.set_yticks(y_pos)
ax_top_neurons.set_yticklabels(labels, fontsize=9)
ax_top_neurons.set_xlabel("")
ax_top_neurons.set_title(
    "Top 3 Most Active Bottleneck Neurons per Digit", fontsize=12, fontweight="bold"
)
ax_top_neurons.set_xlim([0, 1])
ax_top_neurons.set_xticks([])
ax_top_neurons.spines["top"].set_visible(False)
ax_top_neurons.spines["right"].set_visible(False)
ax_top_neurons.spines["bottom"].set_visible(False)

# Row 5: Specialization analysis (moved to row 4 above, this is duplicate - remove)
ax_specialization = fig.add_subplot(gs[4, :])
# For each neuron, find which digit activates it most
neuron_specialization = np.argmax(heatmap_data, axis=0)
specialization_counts = np.bincount(neuron_specialization, minlength=num_classes)

bars = ax_specialization.bar(
    range(num_classes),
    specialization_counts,
    color="steelblue",
    alpha=0.7,
    edgecolor="navy",
)
ax_specialization.set_xlabel("Digit Class", fontsize=11)
ax_specialization.set_ylabel("Number of Specialized Neurons", fontsize=11)
ax_specialization.set_title(
    "Neuron Specialization Distribution\n(Which digit maximally activates each neuron)",
    fontsize=12,
    fontweight="bold",
)
ax_specialization.set_xticks(range(num_classes))
ax_specialization.set_xticklabels([f"{i}" for i in range(num_classes)])
ax_specialization.grid(axis="y", alpha=0.3)

# Add counts on top of bars
for i, (bar, count) in enumerate(zip(bars, specialization_counts)):
    if count > 0:
        ax_specialization.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

plt.suptitle(
    f"Autoencoder Analysis: Effective Filters & Bottleneck Activation Patterns\n"
    f"Model: {model.layers[bottleneck_idx].name} ({bottleneck_dim} dimensions)",
    fontsize=16,
    fontweight="bold",
)

plt.savefig("bottleneck_analysis.png", dpi=150, bbox_inches="tight")
print("✓ Saved to: bottleneck_analysis.png")
plt.show()

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"Bottleneck dimension: {bottleneck_dim}")
print(f"\nNeuron Specialization:")
for digit in range(num_classes):
    count = specialization_counts[digit]
    percentage = (count / bottleneck_dim) * 100
    print(f"  Digit {digit}: {count:2d} neurons ({percentage:5.1f}%) maximally respond")

print(f"\nTop 5 Most Discriminative Neurons:")
# Find neurons with highest variance across digits
neuron_variances = np.var(heatmap_data, axis=0)
top_5_neurons = np.argsort(neuron_variances)[-5:][::-1]
for i, neuron_idx in enumerate(top_5_neurons, 1):
    variance = neuron_variances[neuron_idx]
    max_digit = np.argmax(heatmap_data[:, neuron_idx])
    max_activation = heatmap_data[max_digit, neuron_idx]
    print(
        f"  {i}. Neuron {neuron_idx:2d}: variance={variance:.4f}, "
        f"max for digit {max_digit} (activation={max_activation:.4f})"
    )

print("\n✓ Analysis complete!")
