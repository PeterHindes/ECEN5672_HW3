import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

print("Loading saved EMNIST label-only model...")
model = keras.models.load_model("emnist_autoencoder_label_only.keras")
print("✓ Model loaded successfully!\n")

# Get the decoder part of the model
decoder_input_layer = model.get_layer("combined_bottleneck")
decoder_output = model.get_layer("reconstructed_image").output

# Create a model that takes the combined bottleneck as input
decoder_model = keras.Model(inputs=decoder_input_layer.output, outputs=decoder_output)


# EMNIST ByClass label mapping
# 0-9: digits 0-9
# 10-35: uppercase A-Z
# 36-61: lowercase a-z
def char_to_label(char):
    """Convert character to EMNIST ByClass label"""
    if char.isdigit():
        return int(char)
    elif char.isupper():
        return ord(char) - ord("A") + 10
    elif char.islower():
        return ord(char) - ord("a") + 36
    else:
        return None  # Unsupported character


def generate_character_image(char, variant_values=None):
    """Generate image for a single character with specified variant distribution"""
    label = char_to_label(char)
    if label is None:
        # Return blank image for unsupported characters (like spaces)
        return np.zeros((28, 28), dtype=np.float32)

    # Create one-hot encoded character
    char_bottleneck = np.zeros((1, 62), dtype=np.float32)
    char_bottleneck[0, label] = 1.0

    # Use provided variant values or randomize
    if variant_values is None:
        # Random variant distribution
        variant_values = np.random.dirichlet(np.ones(8), size=1).astype(np.float32)
    else:
        variant_values = np.array(variant_values, dtype=np.float32).reshape(1, -1)
        # Normalize to sum to 1
        variant_sum = np.sum(variant_values)
        if variant_sum > 0:
            variant_values = variant_values / variant_sum
        else:
            variant_values = np.ones((1, 8), dtype=np.float32) / 8.0

    # Combine character and variant
    combined = np.concatenate([char_bottleneck, variant_values], axis=1)

    # Generate image
    generated_image = decoder_model.predict(combined, verbose=0)

    return generated_image[0, :, :, 0]


def render_text(
    text, randomize_variants=True, uniform_variant=None, chars_per_row=None
):
    """
    Render text as a sequence of generated character images

    Args:
        text: String to render
        randomize_variants: If True, use random variants for each character
        uniform_variant: If provided (0-7), use this variant for all characters
        chars_per_row: Number of characters per row (auto if None)
    """
    # Filter to only supported characters and convert to uppercase if needed
    chars = []
    for char in text:
        if char == " ":
            chars.append(" ")  # Keep spaces
        elif char_to_label(char) is not None:
            chars.append(char)
        elif char_to_label(char.upper()) is not None:
            chars.append(char.upper())

    if not chars:
        print("No valid characters to render!")
        return

    num_chars = len(chars)

    # Determine grid layout
    if chars_per_row is None:
        chars_per_row = min(20, num_chars)  # Max 20 chars per row

    num_rows = int(np.ceil(num_chars / chars_per_row))

    print(f"Rendering text: '{text}'")
    print(f"Valid characters: {num_chars}")
    print(f"Layout: {num_rows} rows × {chars_per_row} chars/row")
    print("Generating images...\n")

    # Generate images for each character
    images = []
    for i, char in enumerate(chars):
        if char == " ":
            # Blank space
            images.append(np.zeros((28, 28), dtype=np.float32))
        else:
            # Determine variant
            if uniform_variant is not None:
                # Use specified uniform variant
                variant_values = np.zeros(8, dtype=np.float32)
                variant_values[uniform_variant] = 1.0
            elif randomize_variants:
                # Random variant
                variant_values = None
            else:
                # Equal distribution across all variants
                variant_values = np.ones(8, dtype=np.float32) / 8.0

            img = generate_character_image(char, variant_values)
            images.append(img)

        print(f"Generated {i + 1}/{num_chars}: '{char}'                    ", end="\r")

    print("\n\nCreating visualization...")

    # Create figure
    fig_width = min(chars_per_row * 1.2, 24)
    fig_height = num_rows * 1.2
    fig, axes = plt.subplots(num_rows, chars_per_row, figsize=(fig_width, fig_height))

    # Handle single row/column case
    if num_rows == 1 and chars_per_row == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)
    elif chars_per_row == 1:
        axes = axes.reshape(-1, 1)

    # Plot each character
    for idx, (char, img) in enumerate(zip(chars, images)):
        row = idx // chars_per_row
        col = idx % chars_per_row

        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"'{char}'", fontsize=10, fontweight="bold")
        axes[row, col].axis("off")

    # Hide unused subplots
    for idx in range(num_chars, num_rows * chars_per_row):
        row = idx // chars_per_row
        col = idx % chars_per_row
        axes[row, col].axis("off")

    variant_info = (
        f"Uniform Variant {uniform_variant}"
        if uniform_variant is not None
        else ("Random Variants" if randomize_variants else "Equal Variant Distribution")
    )

    plt.suptitle(
        f'Rendered Text: "{text}"\n{variant_info}',
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


# Interactive text rendering
print("=" * 70)
print("EMNIST Text Renderer")
print("=" * 70)
print("\nThis tool renders text using the EMNIST label-only autoencoder.")
print("Each character is generated from its label + variant distribution.\n")
print("Supported characters:")
print("  - Digits: 0-9")
print("  - Uppercase: A-Z")
print("  - Lowercase: a-z")
print("  - Spaces (rendered as blank)")
print("\n" + "=" * 70)

try:
    while True:
        print("\n")
        text = input("Enter text to render (or 'quit' to exit): ").strip()

        if text.lower() == "quit":
            print("Exiting...")
            break

        if not text:
            print("Please enter some text!")
            continue

        print("\nVariant options:")
        print("  1. Random variants for each character")
        print("  2. Uniform variant (0-7)")
        print("  3. Equal distribution across all variants")

        variant_choice = input("Choose option (1-3, default=1): ").strip()

        randomize = True
        uniform = None

        if variant_choice == "2":
            variant_num = input("Enter variant number (0-7): ").strip()
            try:
                uniform = int(variant_num)
                if uniform < 0 or uniform > 7:
                    print("Invalid variant, using random variants")
                    uniform = None
                else:
                    randomize = False
            except ValueError:
                print("Invalid input, using random variants")
        elif variant_choice == "3":
            randomize = False

        # Ask for custom layout
        layout = input("Characters per row (default=auto, max=20): ").strip()
        chars_per_row = None
        if layout:
            try:
                chars_per_row = int(layout)
                if chars_per_row < 1:
                    chars_per_row = None
            except ValueError:
                print("Invalid input, using auto layout")

        # Render the text
        fig = render_text(
            text,
            randomize_variants=randomize,
            uniform_variant=uniform,
            chars_per_row=chars_per_row,
        )

        if fig:
            # Save option
            save = input("\nSave image? (y/n, default=n): ").strip().lower()
            if save == "y":
                filename = input("Filename (default=rendered_text.png): ").strip()
                if not filename:
                    filename = "rendered_text.png"
                if not filename.endswith(".png"):
                    filename += ".png"
                fig.savefig(filename, dpi=150, bbox_inches="tight")
                print(f"✓ Saved to: {filename}")

            plt.show()

except KeyboardInterrupt:
    print("\n\nExiting...")

print("\nGoodbye!")
