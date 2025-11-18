import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_datasets as tfds

# print("TensorFlow Version:", tf.__version__)
# print("GPU Devices:", tf.config.list_physical_devices('GPU'))

# Define the CNN model
model = Sequential([
    # Encoder
    layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),  # 28→14
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),  # 14→7
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((2, 2)),  # 7→3
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Flatten(),  # 3×3×64 = 576
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu', name='encoder_bottleneck_output'),
    
    # Decoder
    layers.Dense(3136, activation='relu'),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),  # 7→14
    layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),  # 14→28
    layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')
])




# Load the MNIST dataset "Use 10000 training images and 2000 test images from the dataset."
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train[:10000]', 'test[:2000]'],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)

# add normalize and add noise to the input images while keeping the target image unchanged
def normalize_img(data):
    image = tf.cast(data['image'], tf.float32) / 255.
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.3)
    return image + noise, image

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae','mse'],
    jit_compile=True
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
print("\n" + "="*50)
print("Final Evaluation on Test Set:")
print("="*50)
results = model.evaluate(ds_test, verbose=1)
print(f"\nTest Loss (MSE): {results[0]:.6f}")
print(f"Test MAE: {results[1]:.6f}")


model.save('mnist_autoencoder_denoiser_model.keras')
print("\n✓ Full model saved to: mnist_autoencoder_denoiser_model.keras")

