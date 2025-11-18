import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

print("Loading saved model...")
model = keras.models.load_model('mnist_autoencoder_model.keras')
print("✓ Model loaded successfully!\n")

print("Loading test data...")
ds_test = tfds.load(
    'mnist',
    split='test[:2000]',
    shuffle_files=True,
    as_supervised=False,
)

def normalize_img(data):
    image = tf.cast(data['image'], tf.float32) / 255.
    return image

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
print("✓ Test data loaded\n")

print("Press Ctrl+C to exit\n")

try:
    while True:
        # Get a random image
        print("Selecting random image...")
        for images in ds_test.take(1):
            random_idx = np.random.randint(0, images.shape[0])
            test_image = images[random_idx:random_idx+1]
            break

        print("Generating reconstruction...")
        reconstructed = model.predict(test_image, verbose=0)

        # Calculate reconstruction error
        mse = np.mean((test_image.numpy() - reconstructed) ** 2)
        mae = np.mean(np.abs(test_image.numpy() - reconstructed))

        print(f"\nReconstruction Metrics:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}\n")

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(test_image[0].numpy().squeeze(), cmap='gray')
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(reconstructed[0].squeeze(), cmap='gray')
        axes[1].set_title('Reconstructed Image', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle(f'Autoencoder Reconstruction\nMSE: {mse:.6f} | MAE: {mae:.6f}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        print("Opening visualization window...")
        plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")
