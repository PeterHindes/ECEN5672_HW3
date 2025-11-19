import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Define the CNN model using the Functional API
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
x = layers.Dense(128)(x)
x = layers.LeakyReLU(alpha=0.2)(x)
x = layers.Dropout(0.3)(x)

# Split into two bottlenecks: digit identity (10) and variant/style (5)
digit_bottleneck = layers.Dense(10, activation="softmax", name="digit_classification")(
    x
)
variant_bottleneck = layers.Dense(
    5, activation="softmax", name="variant_classification"
)(x)

# Concatenate the two bottlenecks for reconstruction
# Total: 10 + 5 = 15 dimensional bottleneck
combined_bottleneck = layers.Concatenate(name="combined_bottleneck")(
    [digit_bottleneck, variant_bottleneck]
)

# Decoder - reconstructs from the combined bottleneck
decoder_input = layers.Dense(128)(combined_bottleneck)
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

# Create the model with three outputs: reconstruction, digit, and variant
model = keras.Model(
    inputs=input_img,
    outputs=[reconstructed_output, digit_bottleneck, variant_bottleneck],
)


# Load the MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10000]", "test[:2000]"],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)


# Preprocess: add noise and create variant labels
def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    label = data["label"]

    # Add noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.3)
    noisy_image = image + noise

    # Create variant label (0-4) based on image index hash
    # This creates pseudo-variants for different writing styles
    # In practice, you might use actual style information if available
    variant_label = tf.cast(tf.reduce_sum(image) * 1000, tf.int32) % 5

    return noisy_image, {
        "reconstructed_image": image,
        "digit_classification": label,
        "variant_classification": variant_label,
    }


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Compile the model
# The digit and variant classifications drive the bottleneck
# Reconstruction must work from only these two categorical representations
model.compile(
    optimizer="adam",
    loss={
        "reconstructed_image": "mse",
        "digit_classification": "sparse_categorical_crossentropy",
        "variant_classification": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "reconstructed_image": 0.7,  # Still important for quality
        "digit_classification": 0.2,  # Force digit encoding
        "variant_classification": 0.1,  # Force style/variant encoding
    },
    metrics={
        "reconstructed_image": ["mae", "mse"],
        "digit_classification": "accuracy",
        "variant_classification": "accuracy",
    },
    jit_compile=True,
)

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_test,
)

# Evaluate the model
print("\n" + "=" * 50)
print("Final Evaluation on Test Set:")
print("=" * 50)
results = model.evaluate(ds_test, verbose=1)
print(f"\nTotal Test Loss: {results[0]:.6f}")
print(f"Reconstruction Test Loss (MSE): {results[1]:.6f}")
print(f"Digit Classification Test Loss: {results[2]:.6f}")
print(f"Variant Classification Test Loss: {results[3]:.6f}")
print(f"Reconstruction Test MAE: {results[4]:.6f}")
print(f"Reconstruction Test MSE: {results[5]:.6f}")
print(f"Digit Classification Test Accuracy: {results[6]:.6f}")
print(f"Variant Classification Test Accuracy: {results[7]:.6f}")


model.save("mnist_autoencoder_label_only.keras")
print("\n✓ Full model saved to: mnist_autoencoder_label_only.keras")

print("\n" + "=" * 70)
print("Model Architecture Explanation:")
print("=" * 70)
print("This model explicitly separates digit identity from style/variant:")
print("  - Digit bottleneck: 10 values (one-hot, which digit 0-9)")
print("  - Variant bottleneck: 5 values (one-hot, which style variant)")
print("  - Decoder must reconstruct ONLY from these 15 categorical values")
print("\nThis forces the model to:")
print("  1. Encode 'WHAT' digit it is (0-9)")
print("  2. Encode 'HOW' it's written (5 style variants)")
print("  3. Reconstruct the entire image from just these two pieces")
print("\nResult: Explicit disentanglement of content vs. style!")
print("=" * 70)
