import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# ==============================================================================
# ADVERSARIAL STYLE DISENTANGLEMENT VARIANT
# ==============================================================================
# This version adds a "style predictor" that tries to predict character from
# style alone. We PENALIZE accurate predictions (negative loss weight) to
# encourage the style bottleneck to contain ONLY stylistic features.
# ==============================================================================

print("=" * 80)
print("ADVERSARIAL STYLE DISENTANGLEMENT TRAINER")
print("=" * 80)
print("This variant punishes the model if style encodes predictable character info.")
print("Goal: Force style bottleneck to contain ONLY stylistic features, not content.")
print("=" * 80)
print()

# Define the CNN model using the Functional API
# EMNIST images are 28x28, same as MNIST
input_img = keras.Input(shape=(28, 28, 1), name="input_image")

# Encoder - MORE filters for better feature extraction
x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(input_img)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 28→14

x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 14→7

x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 7→3

x = layers.Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Flatten()(x)  # 3×3×128 = 1152

# Dense encoding - larger for lowercase letters
x = layers.Dense(512, kernel_initializer="he_normal")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, kernel_initializer="he_normal")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.2)(x)

# Split into two bottlenecks:
# 1. Character identity (26 lowercase letters only) - supervised
# 2. Style encoding (12) - unsupervised, more dimensions for variety
character_bottleneck = layers.Dense(
    26, activation="softmax", name="character_classification"
)(x)

# Larger style bottleneck for more expressive rendering
style_bottleneck = layers.Dense(12, activation="tanh", name="style_encoding")(x)

# ==============================================================================
# ADVERSARIAL COMPONENT: Style Predictor
# ==============================================================================
# This predictor tries to guess the character from ONLY the style bottleneck.
# We will PUNISH accurate predictions (negative loss weight) to force the
# style bottleneck to be uninformative about character identity.
# ==============================================================================

style_predictor_hidden = layers.Dense(64, kernel_initializer="he_normal")(
    style_bottleneck
)
style_predictor_hidden = layers.LeakyReLU(alpha=0.2)(style_predictor_hidden)
style_predictor_hidden = layers.Dropout(0.3)(style_predictor_hidden)

style_predictor_output = layers.Dense(26, activation="softmax", name="style_predictor")(
    style_predictor_hidden
)

# ==============================================================================

# Concatenate for reconstruction
# Total: 26 + 12 = 38 dimensional bottleneck
combined_bottleneck = layers.Concatenate(name="combined_bottleneck")(
    [character_bottleneck, style_bottleneck]
)

# Decoder - MORE filters for better reconstruction
decoder_input = layers.Dense(512, kernel_initializer="he_normal")(combined_bottleneck)
decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
decoder_input = layers.Dense(3136, kernel_initializer="he_normal")(decoder_input)
decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
decoder_input = layers.Reshape((7, 7, 64))(decoder_input)

x = layers.Conv2DTranspose(
    64, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
)(decoder_input)  # 7→14
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)

x = layers.Conv2DTranspose(
    32, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
)(x)  # 14→28
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU(alpha=0.2)(x)

reconstructed_output = layers.Conv2D(
    1, (3, 3), padding="same", activation="sigmoid", name="reconstructed_image"
)(x)

# Create the model with THREE outputs:
# 1. reconstructed_image - MSE loss
# 2. character_classification - supervised character prediction
# 3. style_predictor - ADVERSARIAL (negative weight)
model = keras.Model(
    inputs=input_img,
    outputs=[reconstructed_output, character_bottleneck, style_predictor_output],
)


# Load the EMNIST dataset (ByClass) and filter for lowercase only
print("Loading EMNIST ByClass dataset (lowercase letters only)...")
print("This dataset will contain 26 classes: a-z (labels 36-61 in ByClass)")
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
print("✓ Dataset loaded\n")


# Filter for lowercase letters only (labels 36-61)
def filter_lowercase(image, label):
    """Keep only lowercase letters (labels 36-61)"""
    return tf.logical_and(label >= 36, label <= 61)


print("Filtering for lowercase letters only...")
ds_train = ds_train.filter(filter_lowercase)
ds_test = ds_test.filter(filter_lowercase)
print("✓ Filtered\n")

# Balance the dataset - equal examples per class
print("Balancing dataset to have equal examples per class...")
NUM_CLASSES = 26  # Only lowercase a-z
EXAMPLES_PER_CLASS_TRAIN = 1500  # More examples per letter: 39,000 total
EXAMPLES_PER_CLASS_TEST = 300  # 7,800 total


def balance_dataset(ds, examples_per_class, num_classes):
    """Sample equal number of examples from each class - optimized version"""
    from collections import defaultdict

    label_groups = defaultdict(list)

    # Stream through dataset efficiently, stop when we have enough
    print("  Streaming and grouping by label...", end="", flush=True)
    total_needed = examples_per_class * num_classes
    total_collected = 0

    for image, label in ds.as_numpy_iterator():
        # Remap labels: 36-61 → 0-25
        remapped_label = label - 36

        # Only collect if we need more of this class
        if len(label_groups[remapped_label]) < examples_per_class:
            label_groups[remapped_label].append((image, remapped_label))
            total_collected += 1

        # Early exit if we have enough of all classes
        if total_collected >= total_needed:
            if all(
                len(label_groups[i]) >= examples_per_class for i in range(num_classes)
            ):
                break

    print(" done!")

    # Sample equal number from each class
    balanced_data = []
    for label in range(num_classes):
        if label in label_groups:
            samples = label_groups[label][:examples_per_class]
            balanced_data.extend(samples)
            char = chr(ord("a") + label)
            print(f"  '{char}' (label {label:2d}): {len(samples):4d} examples")

    # Convert back to dataset
    images = np.array([item[0] for item in balanced_data])
    labels = np.array([item[1] for item in balanced_data])

    return tf.data.Dataset.from_tensor_slices((images, labels))


print("\nBalancing training set:")
ds_train = balance_dataset(ds_train, EXAMPLES_PER_CLASS_TRAIN, NUM_CLASSES)

print("\nBalancing test set:")
ds_test = balance_dataset(ds_test, EXAMPLES_PER_CLASS_TEST, NUM_CLASSES)

print(f"\n✓ Balanced datasets created")
print(
    f"  Training: {NUM_CLASSES} classes × {EXAMPLES_PER_CLASS_TRAIN} examples = {NUM_CLASSES * EXAMPLES_PER_CLASS_TRAIN:,} total"
)
print(
    f"  Test:     {NUM_CLASSES} classes × {EXAMPLES_PER_CLASS_TEST} examples = {NUM_CLASSES * EXAMPLES_PER_CLASS_TEST:,} total"
)
print()


# Preprocess: add noise
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    # Add noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.3)
    noisy_image = image + noise

    return noisy_image, {
        "reconstructed_image": image,
        "character_classification": label,
        "style_predictor": label,  # Same label, but will be punished!
    }


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(NUM_CLASSES * EXAMPLES_PER_CLASS_TRAIN)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# Visualization callback - simple blocking version
class VisualizationCallback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        # Store validation dataset for sampling
        self.val_dataset = validation_data.unbatch()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Sample 3 random examples fresh each epoch
        import random

        # Take a fresh batch and sample from it
        samples_iter = iter(self.val_dataset.shuffle(1000, seed=epoch).take(3))
        samples = list(samples_iter)

        viz_x = np.array([s[0] for s in samples])
        viz_y_labels = np.array([s[1]["character_classification"] for s in samples])

        # Get predictions
        predictions = self.model.predict(viz_x, verbose=0)
        reconstructed = predictions[0]
        char_pred = predictions[1]
        style_pred = predictions[2]

        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        fig.suptitle(f"Epoch {epoch + 1}", fontsize=16, fontweight="bold")

        for i in range(3):
            # Input
            axes[i, 0].imshow(viz_x[i, :, :, 0], cmap="gray")
            axes[i, 0].set_title(f"Input\nTrue: '{chr(ord('a') + viz_y_labels[i])}'")
            axes[i, 0].axis("off")

            # Reconstructed
            axes[i, 1].imshow(reconstructed[i, :, :, 0], cmap="gray")
            char_idx = np.argmax(char_pred[i])
            axes[i, 1].set_title(
                f"Recon\nPred: '{chr(ord('a') + char_idx)}' ({char_pred[i][char_idx]:.1%})"
            )
            axes[i, 1].axis("off")

            # Metrics
            axes[i, 2].axis("off")
            if i == 0:
                metrics_text = f"EPOCH {epoch + 1}\n\n"
                metrics_text += (
                    f"Val Recon MSE: {logs.get('val_reconstructed_image_mse', 0):.6f}\n"
                )
                metrics_text += f"Val Char Acc: {logs.get('val_character_classification_accuracy', 0):.4f}\n"
                metrics_text += f"Val Style Acc: {logs.get('val_style_predictor_accuracy', 0):.4f}\n"
                axes[i, 2].text(
                    0.05,
                    0.95,
                    metrics_text,
                    transform=axes[i, 2].transAxes,
                    fontsize=10,
                    verticalalignment="top",
                    fontfamily="monospace",
                )

        plt.tight_layout()

        # Save to file instead of blocking
        filename = f"training_progress_epoch_{epoch + 1:02d}.png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        print(f"  → Saved visualization to {filename}")

        # Also save as last_epoch.png for auto-refresh
        plt.savefig("last_epoch.png", dpi=100, bbox_inches="tight")

        plt.close(fig)


# Compile the model
# ADVERSARIAL LOSS CONFIGURATION:
# - reconstruction: positive weight (want good reconstruction)
# - character_classification: positive weight (want accurate character prediction)
# - style_predictor: NEGATIVE weight (punish accurate character prediction from style!)

print("=" * 80)
print("LOSS CONFIGURATION:")
print("=" * 80)
print("  reconstructed_image:        +0.6  (84% SSIM + 16% L1)")
print("  character_classification:   +0.4  (maximize character prediction accuracy)")
print("  style_predictor:            -0.15 (MINIMIZE character info in style!)")
print("  + entropy penalty to prevent overconfident predictions")
print("=" * 80)
print("The negative weight creates adversarial training:")
print("  → Model is rewarded when style predictor FAILS")
print("  → This forces style bottleneck to be uninformative about character")
print("  → Reduced weight (-0.15) balances disentanglement and learning")
print("  → Entropy penalty prevents the model from gaming with 100% confidence")
print("=" * 80)
print()


# Custom reconstruction loss: 84% SSIM + 16% L1
def reconstruction_loss(y_true, y_pred):
    # SSIM loss (1 - SSIM so that lower is better)
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    # L1 loss (MAE)
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Combine: 84% SSIM + 16% L1
    return 0.84 * ssim_loss + 0.16 * l1_loss


# Custom loss that adds entropy penalty to style predictor
def style_predictor_loss_with_entropy(y_true, y_pred):
    # Standard cross-entropy loss
    ce_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Entropy penalty: punish overconfident predictions
    # Entropy = -sum(p * log(p)), we want HIGH entropy (uncertain predictions)
    epsilon = 1e-7
    entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + epsilon), axis=-1)
    max_entropy = tf.math.log(26.0)  # Maximum entropy for 26 classes

    # Penalize low entropy (overconfident predictions)
    entropy_penalty = max_entropy - entropy

    # Combine: CE loss + entropy penalty
    return ce_loss + 0.1 * entropy_penalty


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss={
        "reconstructed_image": reconstruction_loss,
        "character_classification": "sparse_categorical_crossentropy",
        "style_predictor": style_predictor_loss_with_entropy,
    },
    loss_weights={
        "reconstructed_image": 0.6,  # Good reconstruction (SSIM + L1)
        "character_classification": 0.4,  # Accurate character prediction
        "style_predictor": -0.15,  # NEGATIVE: punish accurate + overconfident predictions
    },
    metrics={
        "reconstructed_image": ["mae", "mse"],
        "character_classification": "accuracy",
        "style_predictor": "accuracy",  # Low is good!
    },
    jit_compile=False,  # Disable JIT for custom loss functions
)

# Display model architecture
model.summary()

# Create visualization callback
viz_callback = VisualizationCallback(ds_test)

# Train the model
print("\nTraining model with adversarial style disentanglement...")
print("Watch the style_predictor accuracy: lower = better disentanglement!")
print("Visualizations will be saved as training_progress_epoch_XX.png")
print()

history = model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_test,
    callbacks=[viz_callback],
)

# Evaluate the model
print("\n" + "=" * 80)
print("Final Evaluation on Test Set:")
print("=" * 80)
results = model.evaluate(ds_test, verbose=1)
print(f"\nTotal Test Loss: {results[0]:.6f}")
print(f"Reconstruction Test Loss (MSE): {results[1]:.6f}")
print(f"Character Classification Test Loss: {results[2]:.6f}")
print(f"Style Predictor Test Loss: {results[3]:.6f}")
print(f"Reconstruction Test MAE: {results[4]:.6f}")
print(f"Reconstruction Test MSE: {results[5]:.6f}")
print(f"Character Classification Test Accuracy: {results[6]:.6f}")
print(f"Style Predictor Test Accuracy: {results[7]:.6f}  ← LOWER IS BETTER!")

print("\n" + "=" * 80)
print("DISENTANGLEMENT ANALYSIS:")
print("=" * 80)
print(f"Character Classification Accuracy: {results[6] * 100:.2f}%")
print(f"Style Predictor Accuracy:          {results[7] * 100:.2f}%")
print(f"Random Baseline (1/26):            {100 / 26:.2f}%")
print()
if results[7] < 0.10:  # Less than 10% accuracy
    print("✓ EXCELLENT: Style is highly disentangled from character!")
elif results[7] < 0.15:
    print("✓ GOOD: Style shows strong disentanglement.")
elif results[7] < 0.25:
    print("○ MODERATE: Some character info leaks into style.")
else:
    print("✗ POOR: Style still encodes significant character information.")
print("=" * 80)

model.save("emnist_lowercase_adversarial.keras")
print("\n✓ Full model saved to: emnist_lowercase_adversarial.keras")

print("\n" + "=" * 80)
print("Model Architecture Explanation:")
print("=" * 80)
print("Adversarial lowercase EMNIST autoencoder with style disentanglement:")
print("  - Character bottleneck: 26 values (a-z only) - supervised")
print("  - Style bottleneck: 12 continuous values - adversarially regularized")
print("  - Style predictor: MLP that tries to predict character from style")
print("  - NEGATIVE loss weight on style predictor forces disentanglement")
print()
print("Key difference from standard model:")
print("  Standard:    Style can encode any information (including character)")
print("  Adversarial: Style is penalized for encoding character information")
print()
print("Benefits for interactive demo:")
print("  1. Changing style sliders won't accidentally change the letter")
print("  2. Each letter can have consistent style variations")
print("  3. Better transfer of style across different characters")
print("  4. More interpretable and controllable style dimensions")
print("=" * 80)
