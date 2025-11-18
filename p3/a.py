from asyncio.unix_events import BaseChildWatcher

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_datasets.core.visualization import visualizer

# Open the image
img = mpimg.imread("img/lena.png")
height, width, channels = img.shape
# Flatten pixels
pixels = img.ravel()

# If image is float in [0,1], convert to 0-255 integers; otherwise cast and clip
if np.issubdtype(pixels.dtype, np.floating):
    pixels = np.rint(pixels * 255).astype(np.int32)
else:
    pixels = pixels.astype(np.int32)
pixels = np.clip(pixels, 0, 255)

plt.figure()
ax = plt.gca()
# Use explicit integer bin edges so each integer value 0..255 gets its own bin
ax.hist(pixels, bins=range(257), color="gray", rwidth=1.0, align="left")
ax.set_xlabel("Pixel Value (0-255)")
# expand limits slightly so the 0 and 255 bars are not hidden by the axes edges
ax.set_xlim(-0.5, 255.5)
ax.set_xticks(range(0, 256, 32))
ax.set_ylabel("Frequency")
ax.set_title("Image Histogram")

# remove the outline/frame of the plot
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_frame_on(False)
ax.patch.set_visible(False)

plt.show()

# Print the most common pixel value
# print(f"Most common pixel value: {np.bincount(pixels).argmax()}")

# # For each pixel value sorted by frequency
# for value, count in sorted(enumerate(np.bincount(pixels)), key=lambda x: x[1], reverse=True):
#     print(f"Pixel value {value} appears {count} times")

frequencies = sorted(enumerate(np.bincount(pixels)), key=lambda x: x[1], reverse=True)


class Node:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None


def huffman_tree(frequencies):
    nodes = [Node(value, freq) for value, freq in frequencies]
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        new_node = Node(None, left.freq + right.freq)
        new_node.left = left
        new_node.right = right
        nodes.append(new_node)
    return nodes[0]


# Visualize Huffman Tree As an image
import math

from matplotlib import patches


def visualize_huffman_tree(root, figsize=(12, 6), show=True, filename=None):
    """
    Visualize a Huffman tree using matplotlib.

    Parameters
    - root: root Node of the Huffman tree (instances of Node defined above)
    - figsize: figure size tuple
    - show: whether to call plt.show()
    - filename: if provided, save the figure to this path
    Returns (fig, ax)
    """
    if root is None:
        raise ValueError("Root of the Huffman tree is None")

    # Count number of leaves in each subtree (used to allocate vertical depth)
    counts = {}

    def compute_counts(node):
        if node is None:
            return 0
        if node.left is None and node.right is None:
            counts[node] = 1
            return 1
        left = compute_counts(node.left)
        right = compute_counts(node.right)
        counts[node] = left + right
        return counts[node]

    compute_counts(root)

    # Collect leaves in left-to-right order so we can force uniform horizontal spacing
    leaves = []

    def collect_leaves(node):
        if node is None:
            return
        if node.left is None and node.right is None:
            leaves.append(node)
            return
        collect_leaves(node.left)
        collect_leaves(node.right)

    collect_leaves(root)

    num_leaves = len(leaves)
    if num_leaves == 0:
        raise ValueError("Tree contains no leaves")

    # Assign uniform x positions to leaves to force spacing; root/internal nodes get averaged x of children
    leaf_xs = np.linspace(0.0, 1.0, num_leaves) if num_leaves > 1 else np.array([0.5])
    leaf_pos_map = {leaf: x for leaf, x in zip(leaves, leaf_xs)}

    # compute depth (for y coordinate) for each node
    depths = {}

    def compute_depths(node, depth=0):
        if node is None:
            return
        depths[node] = depth
        compute_depths(node.left, depth + 1)
        compute_depths(node.right, depth + 1)

    compute_depths(root)

    # Assign positions: leaves from leaf_pos_map, internal nodes as average of children x; y = -depth
    positions = {}

    def assign_positions(node):
        if node is None:
            return None
        if node.left is None and node.right is None:
            x = float(leaf_pos_map.get(node, 0.5))
            y = -depths.get(node, 0)
            positions[node] = (x, y)
            return x
        # compute children's x
        left_x = assign_positions(node.left)
        right_x = assign_positions(node.right)
        # If one child is missing, use the other's x
        if left_x is None and right_x is None:
            x = 0.5
        elif left_x is None:
            x = right_x
        elif right_x is None:
            x = left_x
        else:
            x = (left_x + right_x) / 2.0
        y = -depths.get(node, 0)
        positions[node] = (x, y)
        return x

    assign_positions(root)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()

    # Draw edges
    for node, (x, y) in positions.items():
        if node.left is not None:
            x2, y2 = positions[node.left]
            ax.plot([x, x2], [y, y2], color="k", linewidth=1)
        if node.right is not None:
            x2, y2 = positions[node.right]
            ax.plot([x, x2], [y, y2], color="k", linewidth=1)

    # Helper to produce bbox styling, highlighting value 0 and 255
    def node_bbox_style(node):
        # defaults
        bbox = dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8)
        fontsize = 8
        fontweight = "normal"
        # Highlight special pixel values
        if node.value is not None:
            try:
                if int(node.value) == 0:
                    bbox = dict(
                        boxstyle="round,pad=0.5", fc="#e0f3ff", ec="#1f78b4", lw=1.4
                    )
                    fontsize = 9
                    fontweight = "bold"
                elif int(node.value) == 255:
                    bbox = dict(
                        boxstyle="round,pad=0.5", fc="#ffe6e6", ec="#b30000", lw=1.4
                    )
                    fontsize = 9
                    fontweight = "bold"
            except Exception:
                # if value is not an int-convertible, leave defaults
                pass
        return bbox, fontsize, fontweight

    # Draw nodes with labels (value and frequency)
    for node, (x, y) in positions.items():
        if node.value is None:
            label = f"{node.freq}"
        else:
            label = f"{node.value}\n{node.freq}"
        bbox, fontsize, fontweight = node_bbox_style(node)
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=fontweight,
            bbox=bbox,
        )

    # Adjust view limits to add some margins and ensure forced spacing is visible
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    if xs and ys:
        # Because we forced leaf spacing on [0,1], add a small margin so edge nodes aren't flush
        xpad = max(0.03, 0.02 * (1.0 / max(1, num_leaves)))
        ypad = 0.6
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    plt.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


root = huffman_tree(frequencies)
visualize_huffman_tree(root, figsize=(14, 6), show=True)


def apply_huffman_encoding(data, root):
    encoding = {}

    def traverse(node, code):
        if node.value is not None:
            encoding[node.value] = code
        else:
            traverse(node.left, code + "0")
            traverse(node.right, code + "1")

    traverse(root, "")
    return "".join(encoding[data[i]] for i in range(len(data)))


encoded = apply_huffman_encoding(pixels, root)
print(encoded)


def decode_huffman(encoded, root):
    decoded = []
    current = root
    for bit in encoded:
        if bit == "0":
            current = current.left
        else:
            current = current.right
        if current.value is not None:
            decoded.append(current.value)
            current = root
    return decoded


decoded = decode_huffman(encoded, root)
# convert back to image
image = np.array(decoded).reshape(height, width, channels)
plt.imshow(image)
plt.show()

# save image
plt.savefig("lena_huffman.png", bbox_inches="tight", dpi=150)

# compute difference to verify it is the original image
difference = np.sum(np.abs(image - img))
print(f"Difference: {difference}")

# display the difference image
plt.imshow(np.abs(image - img))
plt.show()

# def calculate_data_size
