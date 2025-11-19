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

# Split into two bottlenecks:
# 1. Digit identity (10) - supervised with real labels
# 2. Style encoding (5) - unsupervised, no labels needed
digit_bottleneck = layers.Dense(10, activation="softmax", name="digit_classification")(
    x
)

# Unsupervised style encoding - no labels, learns from reconstruction only
style_bottleneck = layers.Dense(5, activation="tanh", name="style_encoding")(x)

# Concatenate for reconstruction
# Total: 10 + 5 = 15 dimensional bottleneck
# Note: No stop_gradient needed since style is unsupervised (no conflicting losses)
combined_bottleneck = layers.Concatenate(name="combined_bottleneck")(
    [digit_bottleneck, style_bottleneck]
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

# Create the model with two outputs: reconstruction and digit classification
model = keras.Model(
    inputs=input_img,
    outputs=[reconstructed_output, digit_bottleneck],
)


# Load the MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10000]", "test[:2000]"],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)


# Preprocess: add noise
def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    label = data["label"]

    # Add noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.3)
    noisy_image = image + noise

    return noisy_image, {
        "reconstructed_image": image,
        "digit_classification": label,
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
# Digit classification uses real labels (supervised)
# Style encoding learns unsupervised from reconstruction only
model.compile(
    optimizer="adam",
    loss={
        "reconstructed_image": "mse",
        "digit_classification": "sparse_categorical_crossentropy",
    },
    loss_weights={
        "reconstructed_image": 0.7,  # Important for quality
        "digit_classification": 0.3,  # Force digit encoding (real labels)
    },
    metrics={
        "reconstructed_image": ["mae", "mse"],
        "digit_classification": "accuracy",
    },
    jit_compile=True,
)

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    ds_train,
    epochs=16,
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
print(f"Reconstruction Test MAE: {results[3]:.6f}")
print(f"Reconstruction Test MSE: {results[4]:.6f}")
print(f"Digit Classification Test Accuracy: {results[5]:.6f}")


model.save("mnist_autoencoder_label_only.keras")
print("\n✓ Full model saved to: mnist_autoencoder_label_only.keras")

print("\n" + "=" * 70)
print("Model Architecture Explanation:")
print("=" * 70)
print("This model explicitly separates digit identity from style:")
print("  - Digit bottleneck: 10 values (one-hot, which digit 0-9)")
print("  - Style bottleneck: 5 continuous values (learned unsupervised)")
print("  - Decoder must reconstruct from digit (categorical) + style (continuous)")
print("\nThis forces the model to:")
print("  1. Encode 'WHAT' digit it is (0-9) - SUPERVISED")
print("  2. Encode 'HOW' it's written (5-dim style) - UNSUPERVISED")
print("  3. Reconstruct the entire image from just these two pieces")
print("\nResult: Explicit disentanglement of content vs. style!")
print("Note: Style is learned purely from reconstruction (no labels needed)")
print("=" * 70)
