import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

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

# Create the model with two outputs: reconstruction and character classification
model = keras.Model(
    inputs=input_img,
    outputs=[reconstructed_output, character_bottleneck],
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
    """Sample equal number of examples from each class"""
    # Convert to list to group by label
    ds_list = list(ds.as_numpy_iterator())

    # Group by label
    from collections import defaultdict

    label_groups = defaultdict(list)
    for image, label in ds_list:
        # Remap labels: 36-61 → 0-25
        remapped_label = label - 36
        label_groups[remapped_label].append((image, remapped_label))

    # Sample equal number from each class
    balanced_data = []
    for label in range(num_classes):
        if label in label_groups:
            # Take first N examples (already shuffled)
            samples = label_groups[label][:examples_per_class]
            balanced_data.extend(samples)
            char = chr(ord("a") + label)
            print(f"  '{char}' (label {label:2d}): {len(samples):4d} examples")

    # Convert back to dataset
    images = [item[0] for item in balanced_data]
    labels = [item[1] for item in balanced_data]

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

# Compile the model
# Character classification uses real labels (supervised)
# Style encoding learns unsupervised from reconstruction only
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss={
        "reconstructed_image": "mse",
        "character_classification": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "reconstructed_image": 0.6,  # Important for quality
        "character_classification": 0.4,  # Force character encoding
    },
    metrics={
        "reconstructed_image": ["mae", "mse"],
        "character_classification": "accuracy",
    },
    jit_compile=True,
)

# Display model architecture
model.summary()

# Train the model
print("\nTraining model...")
history = model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_test,
)

# Evaluate the model
print("\n" + "=" * 50)
print("Final Evaluation on Test Set:")
print("=" * 50)
results = model.evaluate(ds_test, verbose=1)
print(f"\nTotal Test Loss: {results[0]:.6f}")
print(f"Reconstruction Test Loss (MSE): {results[1]:.6f}")
print(f"Character Classification Test Loss: {results[2]:.6f}")
print(f"Reconstruction Test MAE: {results[3]:.6f}")
print(f"Reconstruction Test MSE: {results[4]:.6f}")
print(f"Character Classification Test Accuracy: {results[5]:.6f}")


model.save("emnist_lowercase_autoencoder.keras")
print("\n✓ Full model saved to: emnist_lowercase_autoencoder.keras")

print("\n" + "=" * 70)
print("Model Architecture Explanation:")
print("=" * 70)
print("Lowercase-only EMNIST autoencoder with enhanced capacity:")
print("  - Character bottleneck: 26 values (a-z only)")
print("  - Style bottleneck: 12 continuous values (more expressive)")
print("  - Enhanced encoder: 32→64→64→128 filters (vs original 16→32→16→32)")
print("  - Enhanced decoder: 128→64→32 filters with extra Conv2D layers")
print("  - Batch normalization for stable training")
print("  - He initialization for better gradient flow")
print("\nThis model focuses exclusively on lowercase letters with:")
print("  1. More convolutional filters for better feature extraction")
print("  2. Larger style bottleneck (12 vs 8) for richer variation")
print("  3. More training data per class (1500 vs 1000)")
print("  4. Additional convolutional layers in decoder for finer detail")
print("\nResult: Higher quality lowercase letter rendering!")
print("=" * 70)
