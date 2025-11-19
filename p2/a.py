import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Define the CNN model
model = Sequential(
    [
        # Encoder
        layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            input_shape=(28, 28, 1),
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Flatten(),
        layers.Dense(
            24, kernel_initializer="he_normal", name="encoder_bottleneck_output"
        ),
        layers.LeakyReLU(alpha=0.3),
        # Decoder
        layers.Dense(3136, kernel_initializer="he_normal"),
        layers.LeakyReLU(alpha=0.3),
        layers.Reshape((7, 7, 64)),
        layers.Conv2DTranspose(
            64, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2DTranspose(
            32, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid"),
    ]
)


# Load the MNIST dataset "Use 10000 training images and 2000 test images from the dataset."
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train[:10000]", "test[:2000]"],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)


def normalize_img(data):
    image = tf.cast(data["image"], tf.float32) / 255.0
    # Ensure proper shape (28, 28, 1)
    image = tf.reshape(image, (28, 28, 1))
    return image, image  # Return (input, target) pairs - both are clean


# Normalize images first
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

print(f"Loaded training and test data (clean autoencoder - no noise)")

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"],
    jit_compile=True,
)

# Display model architecture
model.summary()

# Train the model (clean autoencoder - no noise)
print("\nStarting training (clean autoencoder - no noise)...")
print("=" * 50)

history = model.fit(
    ds_train,
    epochs=25,
    validation_data=ds_test,
)

print("\n" + "=" * 50)
print("Training completed!")

# Evaluate the model
print("\n" + "=" * 50)
print("Final Evaluation on Test Set:")
print("=" * 50)
test_loss, test_mae = model.evaluate(ds_test, verbose=1)
print(f"\nTest MSE Loss: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

model.save("mnist_autoencoder_clean_model.keras")
print("\n✓ Full model saved to: mnist_autoencoder_clean_model.keras")

# Check for dead filters in the problematic layer
print("\n" + "=" * 50)
print("Filter Health Check:")
print("=" * 50)

# Get the Conv2D layer with 64 filters (previously 16)
conv_layer = model.layers[12]  # The 4th Conv2D layer
weights = conv_layer.get_weights()[0]  # Shape: (3, 3, 32, 64)

# Calculate the L2 norm of each filter
filter_norms = []
for i in range(weights.shape[-1]):
    filter_weight = weights[:, :, :, i]
    norm = tf.norm(filter_weight).numpy()
    filter_norms.append(norm)

filter_norms = tf.convert_to_tensor(filter_norms)
mean_norm = tf.reduce_mean(filter_norms).numpy()
std_norm = tf.math.reduce_std(filter_norms).numpy()
min_norm = tf.reduce_min(filter_norms).numpy()
max_norm = tf.reduce_max(filter_norms).numpy()

# Count potentially dead filters (very small norms)
dead_threshold = mean_norm * 0.1
dead_filters = tf.reduce_sum(tf.cast(filter_norms < dead_threshold, tf.int32)).numpy()

print(f"\nLayer: {conv_layer.name}")
print(f"Number of filters: {weights.shape[-1]}")
print(f"Mean filter norm: {mean_norm:.6f}")
print(f"Std filter norm: {std_norm:.6f}")
print(f"Min filter norm: {min_norm:.6f}")
print(f"Max filter norm: {max_norm:.6f}")
print(
    f"Potentially dead filters (norm < {dead_threshold:.6f}): {dead_filters}/{weights.shape[-1]}"
)

if dead_filters > 0:
    print(f"\n⚠️  Warning: {dead_filters} filters may be dead or underutilized")
else:
    print(f"\n✓ All filters appear to be active and learning!")
