import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("=" * 80)
print("ANALYZING DEAD/NOISY FILTERS AND PREVENTION TECHNIQUES")
print("=" * 80)
print()

# Load the models
print("Loading models...")
denoising_model = keras.models.load_model("mnist_autoencoder_denoiser_model.keras")
regular_model = keras.models.load_model("mnist_autoencoder_model.keras")
print("✓ Models loaded\n")

# Build models
dummy_input = tf.zeros((1, 28, 28, 1))
_ = denoising_model(dummy_input)
_ = regular_model(dummy_input)


# Find conv layers
def find_last_conv_layer(model):
    conv_layer_idx = None
    for i, layer in enumerate(model.layers):
        if "bottleneck" in layer.name.lower() or isinstance(
            layer, keras.layers.Flatten
        ):
            break
        if isinstance(layer, keras.layers.Conv2D):
            conv_layer_idx = i
    return conv_layer_idx


denoising_conv_idx = find_last_conv_layer(denoising_model)
regular_conv_idx = find_last_conv_layer(regular_model)

# Load test data for activation analysis
print("Loading test data...")
ds_test = tfds.load("mnist", split="test[:1000]", as_supervised=True)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


ds_test = ds_test.map(normalize_img).batch(32)
print("✓ Loaded 1000 test images\n")


# Create feature extraction models
def create_conv_model(model, conv_idx):
    input_tensor = keras.Input(shape=(28, 28, 1))
    output = input_tensor
    for i, layer in enumerate(model.layers):
        output = layer(output)
        if i == conv_idx:
            break
    return keras.Model(inputs=input_tensor, outputs=output)


denoising_conv_model = create_conv_model(denoising_model, denoising_conv_idx)
regular_conv_model = create_conv_model(regular_model, regular_conv_idx)


# Analyze filter activity across the dataset
def analyze_filter_activity(model, dataset, model_name):
    """Analyze which filters are active across the dataset"""
    print(f"Analyzing {model_name} filter activity...")

    all_activations = []
    for images, _ in dataset:
        activations = model.predict(images, verbose=0)
        # Get mean activation per filter for this batch
        mean_per_filter = np.mean(
            activations, axis=(0, 1, 2)
        )  # Average over batch, H, W
        all_activations.append(mean_per_filter)

    # Average across all batches
    avg_activations = np.mean(all_activations, axis=0)
    std_activations = np.std(all_activations, axis=0)

    # Compute variance of each filter's weights
    weights = model.layers[-1].get_weights()[0]  # Get conv layer weights
    filter_weight_variance = np.var(weights, axis=(0, 1, 2))

    return avg_activations, std_activations, filter_weight_variance


denoising_act, denoising_std, denoising_var = analyze_filter_activity(
    denoising_conv_model, ds_test, "Denoising"
)
regular_act, regular_std, regular_var = analyze_filter_activity(
    regular_conv_model, ds_test, "Regular"
)


# Identify "dead" or "noisy" filters
def identify_dead_filters(
    avg_activations, std_activations, weight_variance, threshold=0.01
):
    """
    Identify filters that are likely dead or learning noise
    Criteria:
    - Low mean activation
    - Low activation variance (not responsive to different inputs)
    - Low weight variance (weights are nearly uniform = noise)
    """
    n_filters = len(avg_activations)
    dead_filters = []

    for i in range(n_filters):
        is_dead = False
        reasons = []

        # Very low activation
        if avg_activations[i] < threshold:
            is_dead = True
            reasons.append(f"low_activation({avg_activations[i]:.4f})")

        # Very low activation variance (not responsive)
        if std_activations[i] < threshold:
            is_dead = True
            reasons.append(f"low_variance({std_activations[i]:.4f})")

        # Very low weight variance (uniform = noise)
        if weight_variance[i] < threshold * 0.1:
            is_dead = True
            reasons.append(f"uniform_weights({weight_variance[i]:.6f})")

        if is_dead:
            dead_filters.append((i, reasons))

    return dead_filters


print("\nIdentifying dead/noisy filters...")
denoising_dead = identify_dead_filters(denoising_act, denoising_std, denoising_var)
regular_dead = identify_dead_filters(regular_act, regular_std, regular_var)

print(
    f"\nDenoising model: {len(denoising_dead)} dead/noisy filters out of {len(denoising_act)}"
)
for idx, reasons in denoising_dead[:5]:
    print(f"  Filter {idx}: {', '.join(reasons)}")
if len(denoising_dead) > 5:
    print(f"  ... and {len(denoising_dead) - 5} more")

print(
    f"\nRegular model: {len(regular_dead)} dead/noisy filters out of {len(regular_act)}"
)
for idx, reasons in regular_dead[:5]:
    print(f"  Filter {idx}: {', '.join(reasons)}")
if len(regular_dead) > 5:
    print(f"  ... and {len(regular_dead) - 5} more")

print()

# Create visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

fig.suptitle(
    "Dead/Noisy Filter Analysis",
    fontsize=16,
    fontweight="bold",
)

# Plot 1: Mean activation per filter
ax1 = plt.subplot(gs[0, 0])
x = np.arange(len(denoising_act))
ax1.bar(x, denoising_act, alpha=0.7, color="orange", label="Denoising")
ax1.axhline(y=0.01, color="red", linestyle="--", linewidth=2, label="Dead threshold")
ax1.set_xlabel("Filter Index", fontsize=11)
ax1.set_ylabel("Mean Activation", fontsize=11)
ax1.set_title("Denoising Model: Mean Activation", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Highlight dead filters
for idx, _ in denoising_dead:
    ax1.axvspan(idx - 0.4, idx + 0.4, color="red", alpha=0.2)

ax2 = plt.subplot(gs[0, 1])
ax2.bar(x, regular_act, alpha=0.7, color="green", label="Regular")
ax2.axhline(y=0.01, color="red", linestyle="--", linewidth=2, label="Dead threshold")
ax2.set_xlabel("Filter Index", fontsize=11)
ax2.set_ylabel("Mean Activation", fontsize=11)
ax2.set_title("Regular Model: Mean Activation", fontsize=12, fontweight="bold")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

for idx, _ in regular_dead:
    ax2.axvspan(idx - 0.4, idx + 0.4, color="red", alpha=0.2)

# Plot 2: Activation variance
ax3 = plt.subplot(gs[0, 2])
ax3.bar(x, denoising_std, alpha=0.7, color="orange", label="Denoising")
ax3.bar(x, regular_std, alpha=0.5, color="green", label="Regular")
ax3.axhline(y=0.01, color="red", linestyle="--", linewidth=2, label="Dead threshold")
ax3.set_xlabel("Filter Index", fontsize=11)
ax3.set_ylabel("Activation Std Dev", fontsize=11)
ax3.set_title("Activation Variance (Responsiveness)", fontsize=12, fontweight="bold")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

# Plot 3: Weight variance
ax4 = plt.subplot(gs[1, 0])
ax4.bar(x, denoising_var, alpha=0.7, color="orange", label="Denoising")
ax4.set_xlabel("Filter Index", fontsize=11)
ax4.set_ylabel("Weight Variance", fontsize=11)
ax4.set_title("Denoising: Weight Variance", fontsize=12, fontweight="bold")
ax4.set_yscale("log")
ax4.legend()
ax4.grid(True, alpha=0.3, axis="y")

ax5 = plt.subplot(gs[1, 1])
ax5.bar(x, regular_var, alpha=0.7, color="green", label="Regular")
ax5.set_xlabel("Filter Index", fontsize=11)
ax5.set_ylabel("Weight Variance", fontsize=11)
ax5.set_title("Regular: Weight Variance", fontsize=12, fontweight="bold")
ax5.set_yscale("log")
ax5.legend()
ax5.grid(True, alpha=0.3, axis="y")

# Plot 4: Combined analysis
ax6 = plt.subplot(gs[1, 2])
# Create a 2D scatter: activation vs weight variance
alive_denoising = [
    i for i in range(len(denoising_act)) if i not in [idx for idx, _ in denoising_dead]
]
dead_denoising_idx = [idx for idx, _ in denoising_dead]

ax6.scatter(
    denoising_var[alive_denoising],
    denoising_act[alive_denoising],
    alpha=0.6,
    color="blue",
    s=50,
    label="Active filters",
)
ax6.scatter(
    denoising_var[dead_denoising_idx],
    denoising_act[dead_denoising_idx],
    alpha=0.8,
    color="red",
    s=100,
    marker="x",
    linewidths=3,
    label="Dead/noisy filters",
)
ax6.set_xlabel("Weight Variance (log)", fontsize=11)
ax6.set_ylabel("Mean Activation", fontsize=11)
ax6.set_title("Filter Health Analysis", fontsize=12, fontweight="bold")
ax6.set_xscale("log")
ax6.legend()
ax6.grid(True, alpha=0.3)

# Plot 5: Prevention techniques text
ax7 = plt.subplot(gs[2, :])
ax7.axis("off")

prevention_text = """
PREVENTION TECHNIQUES FOR DEAD/NOISY FILTERS:

1. INITIALIZATION:
   • Use He/Xavier initialization: kernel_initializer='he_normal'
   • Better initial weights help filters start in useful regions

2. ACTIVATION FUNCTIONS:
   • Use LeakyReLU instead of ReLU: activation=LeakyReLU(alpha=0.01)
   • Prevents "dying ReLU" problem where neurons get stuck at zero
   • Or use ELU, SELU for better gradient flow

3. REGULARIZATION:
   • L1/L2 weight regularization: kernel_regularizer=l2(0.01)
   • Activity regularization: activity_regularizer=l1(1e-5)
   • Forces filters to learn meaningful patterns or be penalized

4. BATCH NORMALIZATION:
   • Add BatchNormalization after Conv layers
   • Normalizes activations, prevents saturation, improves gradient flow

5. DROPOUT:
   • Add Dropout layers (0.2-0.5) to prevent over-reliance on specific filters
   • Forces network to use all filters

6. REDUCE FILTER COUNT:
   • Don't use more filters than needed (e.g., try 32 instead of 64)
   • Smaller models are less prone to dead filters

7. LEARNING RATE:
   • Use learning rate warmup and decay
   • Adaptive optimizers like Adam with clipnorm=1.0

8. FILTER PRUNING (Post-training):
   • Remove dead filters after training
   • Fine-tune remaining filters
"""

ax7.text(
    0.05,
    0.5,
    prevention_text,
    fontsize=9,
    verticalalignment="center",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
)

plt.savefig("dead_filter_analysis.png", dpi=150, bbox_inches="tight")
print("✓ Saved to: dead_filter_analysis.png")

# Create example improved model architecture
print("\n" + "=" * 80)
print("EXAMPLE IMPROVED MODEL ARCHITECTURE")
print("=" * 80)
print()

print("Original architecture issues:")
print("  - Uses ReLU (can cause dying neurons)")
print("  - No regularization")
print("  - No batch normalization")
print("  - Possibly too many filters")
print()

print("Improved architecture:")
print()

improved_model = keras.Sequential(
    [
        # Encoder with improvements
        keras.layers.Conv2D(
            16,  # Reduced from 32
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",  # Better initialization
            kernel_regularizer=keras.regularizers.l2(0.01),  # L2 regularization
        ),
        keras.layers.BatchNormalization(),  # Normalize activations
        keras.layers.LeakyReLU(alpha=0.01),  # Prevent dying ReLU
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(
            32,  # Reduced from 64
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(0.01),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(
            32,  # Reduced from 64
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(0.01),
        ),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(
            32,  # Reduced from 64
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(0.01),
            activity_regularizer=keras.regularizers.l1(1e-5),  # Activity regularization
        ),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.01),
        # Bottleneck
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),  # Dropout for robustness
        keras.layers.Dense(64, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Dense(
            32, kernel_initializer="he_normal", name="encoder_bottleneck_output"
        ),
        # Decoder
        keras.layers.Dense(64, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3 * 3 * 32, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Reshape((3, 3, 32)),
        keras.layers.Conv2DTranspose(
            32, (3, 3), strides=2, padding="same", kernel_initializer="he_normal"
        ),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Conv2DTranspose(
            16, (3, 3), strides=2, padding="same", kernel_initializer="he_normal"
        ),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.01),
        keras.layers.Conv2D(1, (3, 3), strides=2, padding="same", activation="sigmoid"),
    ]
)

improved_model.summary()

print("\nKey improvements:")
print("  ✓ Reduced filter count (16→32→32→32 instead of 32→64→64→64)")
print("  ✓ He initialization for better weight initialization")
print("  ✓ LeakyReLU prevents dying neurons")
print("  ✓ Batch normalization after each conv layer")
print("  ✓ L2 regularization on conv weights")
print("  ✓ Activity regularization on bottleneck layer")
print("  ✓ Dropout for robustness")
print()

print("Training tips:")
print("  • Use Adam optimizer with clipnorm=1.0")
print("  • Use learning rate schedule (e.g., ReduceLROnPlateau)")
print("  • Monitor filter usage during training")
print("  • Consider pruning dead filters after initial training")
print()

print("Compilation example:")
print("  model.compile(")
print("      optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),")
print("      loss='mse'")
print("  )")
print()

# Note: Skipping plot_model since pydot is not installed
# To generate model diagram, install: pip install pydot graphviz
print("(Skipped model diagram generation - install pydot to enable)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"Dead/noisy filter statistics:")
print(
    f"  Denoising model: {len(denoising_dead)}/{len(denoising_act)} ({len(denoising_dead) / len(denoising_act) * 100:.1f}%)"
)
print(
    f"  Regular model: {len(regular_dead)}/{len(regular_act)} ({len(regular_dead) / len(regular_act) * 100:.1f}%)"
)
print()
print("Generated files:")
print("  1. dead_filter_analysis.png - Detailed filter activity analysis")
print()
print("=" * 80)

plt.show()
