import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.widgets import RadioButtons, Slider, TextBox
from tensorflow import keras

print("Loading saved EMNIST lowercase-only model...")
model = keras.models.load_model("emnist_lowercase_autoencoder.keras")
print("✓ Model loaded successfully!\n")

# Get the decoder part of the model
decoder_input_layer = model.get_layer("combined_bottleneck")
decoder_output = model.get_layer("reconstructed_image").output

# Create a model that takes the combined bottleneck as input
decoder_model = keras.Model(inputs=decoder_input_layer.output, outputs=decoder_output)


# Lowercase letter mapping (labels 0-25 correspond to a-z)
def char_to_label(char):
    """Convert lowercase character to label (0-25)"""
    if char.islower():
        return ord(char) - ord("a")
    else:
        return None  # Unsupported character


def generate_character_image(char, style_values=None):
    """Generate image for a single lowercase character with specified style"""
    label = char_to_label(char)
    if label is None:
        # Return blank image for unsupported characters (like spaces)
        return np.zeros((28, 28), dtype=np.float32)

    # Create one-hot encoded character
    char_bottleneck = np.zeros((1, 26), dtype=np.float32)
    char_bottleneck[0, label] = 1.0

    # Use provided style values or randomize
    if style_values is None:
        # Random style in [-1, 1] range (tanh activation)
        style_values = np.random.uniform(-1.0, 1.0, size=(1, 12)).astype(np.float32)
    else:
        style_values = np.array(style_values, dtype=np.float32).reshape(1, -1)
        # Clip to tanh range
        style_values = np.clip(style_values, -1.0, 1.0)

    # Combine character and style
    combined = np.concatenate([char_bottleneck, style_values], axis=1)

    # Generate image
    generated_image = decoder_model.predict(combined, verbose=0)

    return generated_image[0, :, :, 0]


# Global state
current_text = "hello world"
current_style_mode = "random"  # "random", "zero", "uniform"
current_uniform_style = np.zeros(12, dtype=np.float32)
random_style_range = 0.5  # How much randomness (0.0 = zero style, 1.0 = full random)


def render_text_concatenated(text):
    """Render text as a single concatenated horizontal image"""
    # Convert to lowercase and filter to only supported characters
    text = text.lower()
    chars = []
    for char in text:
        if char == " ":
            chars.append(" ")  # Keep spaces
        elif char_to_label(char) is not None:
            chars.append(char)

    if not chars:
        return np.ones((28, 280), dtype=np.float32)  # Blank image

    # Generate images for each character
    images = []
    for char in chars:
        if char == " ":
            # Blank space
            images.append(np.zeros((28, 28), dtype=np.float32))
        else:
            # Determine style
            if current_style_mode == "uniform":
                style_values = current_uniform_style.copy()
            elif current_style_mode == "random":
                # Controlled random: blend between zero and random based on slider
                zero_style = np.zeros(12, dtype=np.float32)
                random_style = np.random.uniform(-1.0, 1.0, size=12).astype(np.float32)
                style_values = (
                    1 - random_style_range
                ) * zero_style + random_style_range * random_style
            else:  # zero
                style_values = np.zeros(12, dtype=np.float32)

            img = generate_character_image(char, style_values)
            # Transpose each character image for correct orientation
            images.append(img.T)

    # Concatenate images horizontally
    if images:
        concatenated = np.concatenate(images, axis=1)
    else:
        concatenated = np.zeros((28, 28), dtype=np.float32)

    return concatenated


def update_display():
    """Update the displayed text rendering"""
    print(f"Rendering: '{current_text}' with {current_style_mode} style...")

    rendered_image = render_text_concatenated(current_text)

    ax_image.clear()
    ax_image.imshow(rendered_image, cmap="gray", interpolation="nearest")
    ax_image.set_title(
        f'Rendered Text: "{current_text}"\n'
        f"Style Mode: {current_style_mode.capitalize()}",
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


def on_style_change(label):
    """Handle style mode selection"""
    global current_style_mode
    if label == "Random Style":
        current_style_mode = "random"
    elif label == "Zero Style":
        current_style_mode = "zero"
    elif label == "Uniform Style":
        current_style_mode = "uniform"
    update_display()


def on_slider_change(val):
    """Handle slider changes for uniform style"""
    global current_uniform_style, current_style_mode
    # Update the uniform style vector based on all sliders
    for i in range(12):
        current_uniform_style[i] = style_sliders[i].val
    if current_style_mode == "uniform":
        update_display()


def on_randomness_slider_change(val):
    """Handle random style range slider change"""
    global random_style_range
    random_style_range = val
    if current_style_mode == "random":
        update_display()


# Create figure and layout
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(
    6, 4, height_ratios=[0.4, 2.5, 0.8, 0.15, 0.15, 0.15], hspace=0.5, wspace=0.4
)

# Title
fig.text(
    0.5,
    0.97,
    "Interactive EMNIST Lowercase Text Renderer",
    ha="center",
    fontsize=18,
    fontweight="bold",
)

# Text input box (top)
ax_textbox = fig.add_subplot(gs[0, :])
ax_textbox.axis("off")
textbox_axes = plt.axes([0.15, 0.91, 0.7, 0.04])
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

# Style mode radio buttons (left side)
ax_radio_style = fig.add_subplot(gs[2, 0])
ax_radio_style.set_title("Style Mode", fontsize=12, fontweight="bold")
radio_style = RadioButtons(
    ax_radio_style,
    labels=["Random Style", "Zero Style", "Uniform Style"],
    active=0,
)
radio_style.on_clicked(on_style_change)

# Randomness slider (for random mode)
ax_random_slider = fig.add_subplot(gs[2, 1:])
ax_random_slider.set_title(
    "Random Style Range (for Random mode)", fontsize=12, fontweight="bold"
)
randomness_slider_axes = plt.axes([0.45, 0.38, 0.5, 0.02])
randomness_slider = Slider(
    randomness_slider_axes,
    "Randomness",
    0.0,
    1.0,
    valinit=random_style_range,
    valstep=0.01,
)
randomness_slider.on_changed(on_randomness_slider_change)
ax_random_slider.axis("off")

# Style dimension sliders (for uniform mode)
style_sliders = []
slider_titles = fig.add_subplot(gs[3, :])
slider_titles.text(
    0.5,
    0.5,
    "Uniform Style Controls (12 dimensions, -1.0 to 1.0)",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
)
slider_titles.axis("off")

# Create 12 sliders in 3 rows x 4 columns
for i in range(12):
    row = 4 + (i // 4)
    col = i % 4
    # Position calculation
    left = 0.08 + col * 0.23
    bottom = 0.16 - (i // 4) * 0.05
    width = 0.18
    height = 0.02

    slider_ax = plt.axes([left, bottom, width, height])
    slider = Slider(
        slider_ax,
        f"S{i}",
        -1.0,
        1.0,
        valinit=0.0,
        valstep=0.05,
    )
    slider.on_changed(on_slider_change)
    style_sliders.append(slider)

# Instructions (bottom)
instructions_text = (
    "Instructions:\n"
    "• Type lowercase text in the box above and press Enter (uppercase will be converted to lowercase)\n"
    "• Choose style mode: Random (controlled by slider), Zero (neutral style), or Uniform (custom style)\n"
    "• Random Style Range: 0.0 = neutral, 1.0 = full randomness per character\n"
    "• Uniform Style: Adjust 12 sliders (S0-S11) to create a consistent custom style\n"
    "• Supported: lowercase letters (a-z) and spaces"
)

fig.text(
    0.5,
    0.01,
    instructions_text,
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
)

# Initial display
update_display()

print("\n" + "=" * 70)
print("Interactive Lowercase Text Renderer Ready!")
print("=" * 70)
print("\nInstructions:")
print("  1. Type text in the text box at the top (lowercase only)")
print("  2. Press Enter to render")
print("  3. Choose style mode with radio buttons")
print("  4. Random mode: Adjust randomness slider (0.0 = neutral, 1.0 = varied)")
print("  5. Uniform mode: Adjust 12 style sliders to create custom styling")
print("  6. Characters are displayed side-by-side in a single image")
print("\nStyle Dimensions (S0-S11):")
print("  Each dimension controls different aspects of letter appearance")
print("  Experiment with values from -1.0 to 1.0 to see effects")
print("\nClose the window to exit.")
print("=" * 70)

plt.show()
