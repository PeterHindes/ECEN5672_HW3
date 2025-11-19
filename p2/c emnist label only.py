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
x = layers.Dense(128)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.3)(x)

# Split into two bottlenecks: character identity (62) and variant/style (8)
# EMNIST ByClass has 62 classes: 10 digits + 26 uppercase + 26 lowercase
character_bottleneck = layers.Dense(
    62, activation="softmax", name="character_classification"
)(x)
variant_bottleneck = layers.Dense(
    8, activation="softmax", name="variant_classification"
)(x)

# Concatenate the two bottlenecks for reconstruction
# Total: 62 + 8 = 70 dimensional bottleneck
combined_bottleneck = layers.Concatenate(name="combined_bottleneck")(
    [character_bottleneck, variant_bottleneck]
)

# Decoder - reconstructs from the combined bottleneck
decoder_input = layers.Dense(128)(combined_bottleneck)
decoder_input = layers.LeakyReLU(alpha=0.2)(decoder_input)
decoder_input = layers.Dense(256)(decoder_input)
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

# Create the model with three outputs: reconstruction, character, and variant
model = keras.Model(
    inputs=input_img,
    outputs=[reconstructed_output, character_bottleneck, variant_bottleneck],
)


# Load the EMNIST dataset (ByClass)
print("Loading EMNIST ByClass dataset...")
print("This dataset contains 62 classes: 10 digits + 26 uppercase + 26 lowercase")
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/byclass",
    split=["train[:20000]", "test[:4000]"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
print("✓ Dataset loaded\n")


# Preprocess: add noise and create variant labels
def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    # Add noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.3)
    noisy_image = image + noise

    # Create variant label (0-7) based on image intensity hash
    # This creates pseudo-variants for different writing styles
    variant_label = tf.cast(tf.reduce_sum(image) * 1000, tf.int32) % 8

    return noisy_image, {
        "reconstructed_image": image,
        "character_classification": label,
        "variant_classification": variant_label,
    }


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(20000)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Compile the model
# The character and variant classifications drive the bottleneck
# Reconstruction must work from only these two categorical representations
model.compile(
    optimizer="adam",
    loss={
        "reconstructed_image": "mse",
        "character_classification": "sparse_categorical_crossentropy",
        "variant_classification": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "reconstructed_image": 0.7,  # Still important for quality
        "character_classification": 0.25,  # Force character encoding
        "variant_classification": 0.05,  # Force style/variant encoding
    },
    metrics={
        "reconstructed_image": ["mae", "mse"],
        "character_classification": "accuracy",
        "variant_classification": "accuracy",
    },
    jit_compile=True,
)

# Display model architecture
model.summary()

# Train the model
print("\nTraining model...")
history = model.fit(
    ds_train,
    epochs=30,
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
print(f"Variant Classification Test Loss: {results[3]:.6f}")
print(f"Reconstruction Test MAE: {results[4]:.6f}")
print(f"Reconstruction Test MSE: {results[5]:.6f}")
print(f"Character Classification Test Accuracy: {results[6]:.6f}")
print(f"Variant Classification Test Accuracy: {results[7]:.6f}")


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
print("  - Variant bottleneck: 8 values (one-hot, which style variant)")
print("  - Decoder must reconstruct ONLY from these 70 categorical values")
print("\nThis forces the model to:")
print("  1. Encode 'WHAT' character it is (62 classes)")
print("  2. Encode 'HOW' it's written (8 style variants)")
print("  3. Reconstruct the entire image from just these two pieces")
print("\nResult: Explicit disentanglement of content vs. style!")
print("=" * 70)
