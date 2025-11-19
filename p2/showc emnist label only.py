import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

print("Loading saved EMNIST label-only denoising model...")
model = keras.models.load_model("emnist_autoencoder_label_only.keras")
print("✓ Model loaded successfully!\n")

print("Loading test data...")
ds_test = tfds.load(
    "emnist/byclass",
    split="test[:4000]",
    shuffle_files=True,
    as_supervised=True,
)


# EMNIST ByClass label mapping
# 0-9: digits 0-9
# 10-35: uppercase A-Z
# 36-61: lowercase a-z
def label_to_char(label):
    """Convert EMNIST ByClass label to character"""
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(ord("A") + label - 10)
    else:
        return chr(ord("a") + label - 36)


def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
print("✓ Test data loaded\n")

print("Press Ctrl+C to exit\n")

try:
    while True:
        # Get a random image and its label
        print("Selecting random image...")
        for images, labels in ds_test.take(1):
            random_idx = np.random.randint(0, images.shape[0])
            clean_image = images[random_idx : random_idx + 1]
            true_label = labels[random_idx].numpy()
            break

        true_char = label_to_char(true_label)

        # Add noise to the image
        noise = tf.random.normal(shape=tf.shape(clean_image), mean=0.0, stddev=0.3)
        noisy_image = clean_image + noise

        print("Generating denoised reconstruction and predictions...")
        denoised_image, char_probs, variant_probs = model.predict(
            noisy_image, verbose=0
        )

        predicted_label = np.argmax(char_probs[0])
        predicted_char = label_to_char(predicted_label)
        predicted_variant = np.argmax(variant_probs[0])

        # Get confidence scores
        char_confidence = char_probs[0][predicted_label]
        variant_confidence = variant_probs[0][predicted_variant]

        # Calculate reconstruction error (comparing to clean image)
        mse = np.mean((clean_image.numpy() - denoised_image) ** 2)
        mae = np.mean(np.abs(clean_image.numpy() - denoised_image))

        # Calculate noise level
        noise_mse = np.mean((clean_image.numpy() - noisy_image.numpy()) ** 2)

        print(f"\nReconstruction Metrics:")
        print(f"  Noise MSE:     {noise_mse:.6f}")
        print(f"  Denoised MSE:  {mse:.6f}")
        print(f"  Denoised MAE:  {mae:.6f}")
        print(f"  Improvement:   {((noise_mse - mse) / noise_mse * 100):.2f}%")
        print(f"\nCharacter Classification:")
        print(f"  True Character:      '{true_char}' (label {true_label})")
        print(f"  Predicted Character: '{predicted_char}' (label {predicted_label})")
        print(f"  Confidence:          {char_confidence:.2%}")
        print(f"  Correct:             {'✓' if true_label == predicted_label else '✗'}")
        print(f"\nVariant Classification:")
        print(f"  Predicted Variant: {predicted_variant}")
        print(f"  Confidence:        {variant_confidence:.2%}")

        # Display top 5 character probabilities
        print(f"\nTop 5 Character Predictions:")
        top_5_chars = np.argsort(char_probs[0])[-5:][::-1]
        for rank, char_idx in enumerate(top_5_chars, 1):
            prob = char_probs[0][char_idx]
            char = label_to_char(char_idx)
            marker = "★" if char_idx == true_label else " "
            print(f"  {rank}. '{char}' (label {char_idx:2d}): {prob:.2%} {marker}")

        # Create 2-row visualization
        # Top row: Clean | Noisy | Denoised
        # Bottom row: Character Probabilities | Variant Probabilities
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

        # Top Row - Images
        # Panel 1: Clean Original
        ax0 = fig.add_subplot(gs[0, 0])
        # Transpose for correct display (EMNIST images are stored rotated)
        ax0.imshow(clean_image[0].numpy().squeeze().T, cmap="gray")
        ax0.set_title(
            f"Clean Original\n(True: '{true_char}' label {true_label})",
            fontsize=14,
            fontweight="bold",
        )
        ax0.axis("off")

        # Panel 2: Noisy Input
        ax1 = fig.add_subplot(gs[0, 1])
        # Transpose for correct display (EMNIST images are stored rotated)
        ax1.imshow(noisy_image[0].numpy().squeeze().T, cmap="gray")
        ax1.set_title(
            f"Noisy Input\n(MSE: {noise_mse:.4f})", fontsize=14, fontweight="bold"
        )
        ax1.axis("off")

        # Panel 3: Denoised Output
        ax2 = fig.add_subplot(gs[0, 2])
        # Transpose for correct display (EMNIST images are stored rotated)
        ax2.imshow(denoised_image[0].squeeze().T, cmap="gray")
        ax2.set_title(
            f"Denoised Output\n(Predicted: '{predicted_char}', MSE: {mse:.4f})",
            fontsize=14,
            fontweight="bold",
        )
        ax2.axis("off")

        # Bottom Row - Probability Distributions
        # Character probabilities (left side of bottom row)
        char_subplot = fig.add_subplot(gs[1, 0:2])

        # Show top 20 character probabilities for readability
        top_20_indices = np.argsort(char_probs[0])[-20:]
        top_20_probs = char_probs[0][top_20_indices]
        top_20_labels = [label_to_char(idx) for idx in top_20_indices]

        char_bars = char_subplot.barh(
            range(20), top_20_probs, color="steelblue", alpha=0.7
        )

        # Highlight predicted and true
        for i, idx in enumerate(top_20_indices):
            if idx == predicted_label:
                char_bars[i].set_color("darkgreen")
                char_bars[i].set_alpha(1.0)
            if idx == true_label and idx != predicted_label:
                char_bars[i].set_color("darkred")
                char_bars[i].set_alpha(0.8)

        char_subplot.set_yticks(range(20))
        char_subplot.set_yticklabels([f"'{c}'" for c in top_20_labels])
        char_subplot.set_xlabel("Probability", fontsize=11)
        char_subplot.set_ylabel("Character", fontsize=11)
        char_subplot.set_title(
            f"Top 20 Character Probabilities (Predicted: '{predicted_char}', Confidence: {char_confidence:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        char_subplot.set_xlim([0, 1])
        char_subplot.grid(axis="x", alpha=0.3)

        # Variant probabilities (right side of bottom row)
        variant_subplot = fig.add_subplot(gs[1, 2])
        variant_bars = variant_subplot.bar(
            range(8), variant_probs[0], color="coral", alpha=0.7
        )
        variant_bars[predicted_variant].set_color("darkviolet")
        variant_bars[predicted_variant].set_alpha(1.0)
        variant_subplot.set_xlabel("Variant", fontsize=11)
        variant_subplot.set_ylabel("Probability", fontsize=11)
        variant_subplot.set_title(
            f"Variant Probabilities\n(Predicted: {predicted_variant}, Confidence: {variant_confidence:.1%})",
            fontsize=12,
            fontweight="bold",
        )
        variant_subplot.set_xticks(range(8))
        variant_subplot.set_ylim([0, 1])
        variant_subplot.grid(axis="y", alpha=0.3)

        improvement = (noise_mse - mse) / noise_mse * 100
        correct_prediction = (
            "✓ Correct" if true_label == predicted_label else "✗ Incorrect"
        )

        plt.suptitle(
            f"EMNIST Label-Only Denoising Autoencoder ({correct_prediction})\n"
            f"Noise Reduction: {improvement:.2f}% | "
            f"True: '{true_char}' → Predicted: '{predicted_char}' (conf: {char_confidence:.1%}), Variant={predicted_variant} (conf: {variant_confidence:.1%})",
            fontsize=16,
            fontweight="bold",
        )

        print("\nOpening visualization window...")
        plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")
