import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import RadioButtons, Slider, TextBox
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


# Global state
current_text = "Hello World"
current_variant_mode = "random"  # "random", "uniform", "equal"
current_uniform_variant = 0
random_variant_range = 0.5  # How much randomness (0.0 = all equal, 1.0 = full random)


def render_text_concatenated(text):
    """Render text as a single concatenated horizontal image"""
    # Filter to only supported characters
    chars = []
    for char in text:
        if char == " ":
            chars.append(" ")  # Keep spaces
        elif char_to_label(char) is not None:
            chars.append(char)
        elif char_to_label(char.upper()) is not None:
            chars.append(char.upper())

    if not chars:
        return np.ones((28, 280), dtype=np.float32)  # Blank image

    # Generate images for each character
    images = []
    for char in chars:
        if char == " ":
            # Blank space
            images.append(np.zeros((28, 28), dtype=np.float32))
        else:
            # Determine variant
            if current_variant_mode == "uniform":
                variant_values = np.zeros(8, dtype=np.float32)
                variant_values[current_uniform_variant] = 1.0
            elif current_variant_mode == "random":
                # Controlled random: blend between equal and random based on slider
                equal_dist = np.ones(8, dtype=np.float32) / 8.0
                random_dist = np.random.dirichlet(np.ones(8)).astype(np.float32)
                variant_values = (
                    1 - random_variant_range
                ) * equal_dist + random_variant_range * random_dist
            else:  # equal
                variant_values = np.ones(8, dtype=np.float32) / 8.0

            img = generate_character_image(char, variant_values)
            # Transpose each character image immediately for correct orientation
            images.append(img.T)

    # Concatenate images horizontally (each image is now 28x28 after transpose)
    # Stack along width dimension (axis=1)
    if images:
        concatenated = np.concatenate(images, axis=1)
    else:
        concatenated = np.zeros((28, 28), dtype=np.float32)

    return concatenated


def update_display():
    """Update the displayed text rendering"""
    print(f"Rendering: '{current_text}' with {current_variant_mode} variants...")

    rendered_image = render_text_concatenated(current_text)

    ax_image.clear()
    # Images are already transposed during concatenation
    ax_image.imshow(rendered_image, cmap="gray", interpolation="nearest")
    ax_image.set_title(
        f'Rendered Text: "{current_text}"\n'
        f"Variant Mode: {current_variant_mode.capitalize()}"
        + (
            f" (variant {current_uniform_variant})"
            if current_variant_mode == "uniform"
            else ""
        ),
        fontsize=14,
        fontweight="bold",
    )
    ax_image.axis("off")

    plt.draw()


def on_text_submit(text):
    """Handle text input"""
    global current_text
    current_text = text
    update_display()


def on_variant_change(label):
    """Handle variant mode selection"""
    global current_variant_mode
    if label == "Random Variants":
        current_variant_mode = "random"
    elif label == "Equal Distribution":
        current_variant_mode = "equal"
    update_display()


def on_uniform_variant_change(label):
    """Handle uniform variant selection"""
    global current_variant_mode, current_uniform_variant
    current_variant_mode = "uniform"
    current_uniform_variant = int(label)
    # Update radio button
    radio_variant.set_active(2)  # Set to "Uniform Variant"
    update_display()


def on_slider_change(val):
    """Handle random variant range slider change"""
    global random_variant_range
    random_variant_range = val
    if current_variant_mode == "random":
        update_display()


# Create figure and layout
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(5, 2, height_ratios=[0.5, 3, 1, 1, 0.5], hspace=0.4, wspace=0.3)

# Title
fig.text(
    0.5,
    0.95,
    "Interactive EMNIST Text Renderer",
    ha="center",
    fontsize=18,
    fontweight="bold",
)

# Text input box (top)
ax_textbox = fig.add_subplot(gs[0, :])
ax_textbox.axis("off")
textbox_axes = plt.axes([0.15, 0.88, 0.7, 0.05])
text_box = TextBox(
    textbox_axes,
    "Enter Text:",
    initial=current_text,
    color="lightgray",
    hovercolor="lightblue",
)
text_box.on_submit(on_text_submit)

# Image display (middle)
ax_image = fig.add_subplot(gs[1, :])
ax_image.set_title("Rendered Text", fontsize=14, fontweight="bold")
ax_image.axis("off")

# Variant mode radio buttons (bottom left)
ax_radio_variant = fig.add_subplot(gs[2, 0])
ax_radio_variant.set_title("Variant Mode", fontsize=12, fontweight="bold")
radio_variant = RadioButtons(
    ax_radio_variant,
    labels=["Random Variants", "Equal Distribution", "Uniform Variant"],
    active=0,
)
radio_variant.on_clicked(on_variant_change)

# Uniform variant selector (bottom right)
ax_radio_uniform = fig.add_subplot(gs[2, 1])
ax_radio_uniform.set_title(
    "Select Uniform Variant (0-7)", fontsize=12, fontweight="bold"
)
radio_uniform = RadioButtons(
    ax_radio_uniform, labels=["0", "1", "2", "3", "4", "5", "6", "7"], active=0
)
radio_uniform.on_clicked(on_uniform_variant_change)

# Random variant slider (bottom)
ax_slider = fig.add_subplot(gs[3, :])
ax_slider.set_title(
    "Random Variant Range (for Random mode)", fontsize=12, fontweight="bold"
)
slider_axes = plt.axes([0.2, 0.18, 0.6, 0.03])
variant_slider = Slider(
    slider_axes,
    "Randomness",
    0.0,
    1.0,
    valinit=random_variant_range,
    valstep=0.01,
)
variant_slider.on_changed(on_slider_change)
ax_slider.axis("off")

# Instructions (bottom)
instructions_text = (
    "Instructions:\n"
    "• Type text in the box above and press Enter\n"
    "• Choose variant mode: Random (controlled by slider below), Equal (average), or Uniform (same variant for all)\n"
    "• Random Variant Range: 0.0 = no variation, 1.0 = full randomness per character\n"
    "• If using Uniform, select which variant (0-7) on the right\n"
    "• Supported: digits (0-9), uppercase (A-Z), lowercase (a-z), spaces"
)

fig.text(
    0.5,
    0.02,
    instructions_text,
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Initial display
update_display()

print("\n" + "=" * 70)
print("Interactive Text Renderer Ready!")
print("=" * 70)
print("\nInstructions:")
print("  1. Type text in the text box at the top")
print("  2. Press Enter to render")
print("  3. Choose variant mode with radio buttons")
print("  4. Adjust random variant slider (0.0 = consistent, 1.0 = varied)")
print("  5. Characters are displayed side-by-side in a single image")
print("\nClose the window to exit.")
print("=" * 70)

plt.show()
