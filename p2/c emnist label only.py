import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Define the CNN model using the Functional API
# EMNIST images are 28x28, same as MNIST
input_img = keras.Input(shape=(28, 28, 1), name="input_image")

# Encoder
x = layers.Conv2D(16, (3, 3), padding="same")(input_img)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 28→14
x = layers.Conv2D(32, (3, 3), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 14→7
x = layers.Conv2D(16, (3, 3), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)  # 7→3
x = layers.Conv2D(32, (3, 3), padding="same")(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Flatten()(x)  # 3×3×32 = 288

# Dense encoding
x = layers.Dense(256)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.3)(x)
# x = layers.Dense(128)(x)
# x = layers.LeakyReLU(alpha=0.2)(x)
# x = layers.Dropout(0.3)(x)

# Split into two bottlenecks:
# 1. Character identity (62) - supervised with real labels
# 2. Style encoding (8) - unsupervised, no labels needed
# EMNIST ByClass has 62 classes: 10 digits + 26 uppercase + 26 lowercase
character_bottleneck = layers.Dense(
    62, activation="softmax", name="character_classification"
)(x)

# Unsupervised style encoding - no labels, learns from reconstruction only
style_bottleneck = layers.Dense(8, activation="tanh", name="style_encoding")(x)

# Concatenate for reconstruction
# Total: 62 + 8 = 70 dimensional bottleneck
# Note: No stop_gradient needed since style is unsupervised (no conflicting losses)
combined_bottleneck = layers.Concatenate(name="combined_bottleneck")(
    [character_bottleneck, style_bottleneck]
)

# Decoder - reconstructs from the combined bottleneck
# decoder_input = layers.Dense(128)(combined_bottleneck)
# decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
# decoder_input = layers.Dense(256)(decoder_input)
decoder_input = layers.Dense(256)(combined_bottleneck)
decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
decoder_input = layers.Dense(3136)(decoder_input)
decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
decoder_input = layers.Reshape((7, 7, 64))(decoder_input)

x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(
    decoder_input
)  # 7→14
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(x)  # 14→28
x = layers.LeakyReLU(alpha=0.2)(x)
reconstructed_output = layers.Conv2D(
    1, (3, 3), padding="same", activation="sigmoid", name="reconstructed_image"
)(x)

# Create the model with two outputs: reconstruction and character classification
model = keras.Model(
    inputs=input_img,
    outputs=[reconstructed_output, character_bottleneck],
)


# Load the EMNIST dataset (ByClass)
print("Loading EMNIST ByClass dataset...")
print("This dataset contains 62 classes: 10 digits + 26 uppercase + 26 lowercase")
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
print("✓ Dataset loaded\n")

# Balance the dataset - equal examples per class
print("Balancing dataset to have equal examples per class...")
NUM_CLASSES = 62
EXAMPLES_PER_CLASS_TRAIN = 1000  # 62,000 total
EXAMPLES_PER_CLASS_TEST = 200  # 12,400 total


def balance_dataset(ds, examples_per_class, num_classes):
    """Sample equal number of examples from each class"""
    # Convert to list to group by label
    ds_list = list(ds.as_numpy_iterator())

    # Group by label
    from collections import defaultdict

    label_groups = defaultdict(list)
    for image, label in ds_list:
        label_groups[label].append((image, label))

    # Sample equal number from each class
    balanced_data = []
    for label in range(num_classes):
        if label in label_groups:
            # Take first N examples (already shuffled)
            samples = label_groups[label][:examples_per_class]
            balanced_data.extend(samples)
            print(f"  Class {label:2d}: {len(samples):4d} examples")

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
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss={
        "reconstructed_image": "mse",
        "character_classification": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "reconstructed_image": 0.6,  # Important for quality
        "character_classification": 0.4,  # Force character encoding (real labels)
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
    epochs=13,
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


model.save("emnist_autoencoder_label_only.keras")
print("\n✓ Full model saved to: emnist_autoencoder_label_only.keras")

print("\n" + "=" * 70)
print("Model Architecture Explanation:")
print("=" * 70)
print("This model explicitly separates character identity from style/variant:")
print("  - Character bottleneck: 62 values (one-hot)")
print("    * 0-9: Digits")
print("    * 10-35: Uppercase letters A-Z")
print("    * 36-61: Lowercase letters a-z")
print("  - Style bottleneck: 8 continuous values (learned unsupervised)")
print("  - Decoder must reconstruct from character (categorical) + style (continuous)")
print("\nThis forces the model to:")
print("  1. Encode 'WHAT' character it is (62 classes) - SUPERVISED")
print("  2. Encode 'HOW' it's written (8-dim style) - UNSUPERVISED")
print("  3. Reconstruct the entire image from just these two pieces")
print("\nResult: Explicit disentanglement of content vs. style!")
print("Note: Style is learned purely from reconstruction (no labels needed)")
print("=" * 70)
