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
        layers.MaxPooling2D((2, 2)),  # 28→14
        layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),  # 14→7
        layers.Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.MaxPooling2D((2, 2)),  # 7→3
        layers.Conv2D(16, (3, 3), padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),
        layers.Flatten(),  # 3×3×64 = 576
        layers.Dense(128, kernel_initializer="he_normal"),
        layers.LeakyReLU(alpha=0.3),
        layers.Dropout(0.2),
        layers.Dense(
            32, kernel_initializer="he_normal", name="encoder_bottleneck_output"
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
        layers.LeakyReLU(alpha=0.3),  # 7→14
        layers.Conv2DTranspose(
            32, (3, 3), strides=(2, 2), padding="same", kernel_initializer="he_normal"
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.3),  # 14→28
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
    return image, image


ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Compile the model with lower learning rate to prevent dead filters
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="mse",
    metrics=["mae"],
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
test_loss, test_mae = model.evaluate(ds_test, verbose=1)
print(f"\nTest MSE Loss: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

model.save("mnist_autoencoder_model.keras")
print("\n✓ Full model saved to: mnist_autoencoder_model.keras")
