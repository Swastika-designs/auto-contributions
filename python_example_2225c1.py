# Learning Objective:
# This tutorial teaches how to generate abstract, procedural pixel art by
# mapping mathematical functions directly to pixel colors using Python's
# Pillow library. You will learn to:
# 1. Understand how pixel coordinates can be transformed for mathematical input.
# 2. Apply mathematical functions to normalized coordinates to produce a value.
# 3. Convert this numerical value into an RGB color for a pixel.
# 4. Use the Pillow library to create and save the generated image.

from PIL import Image
import math
from typing import Callable, Tuple

def map_value_to_rgb(value: float, min_val: float, max_val: float) -> Tuple[int, int, int]:
    """
    Converts a numerical value within a given range to an RGB color tuple.
    This function scales the input 'value' from its 'min_val' to 'max_val'
    range into a color. It's the core of how abstract values become visible colors.

    Args:
        value (float): The input numerical value, typically the result of a math function.
        min_val (float): The minimum possible value the math function can return.
        max_val (float): The maximum possible value the math function can return.

    Returns:
        Tuple[int, int, int]: An RGB color tuple (0-255 for each component).
    """
    # Normalize the value to a 0-1 range first.
    # WHY: This makes it easier to map to different color scales consistently,
    # regardless of the specific min/max output of the math function.
    normalized_value = (value - min_val) / (max_val - min_val)

    # Clamp the normalized value to ensure it stays within [0, 1]
    # WHY: This prevents issues (like index out of bounds or negative color values)
    # if the input 'value' somehow exceeds 'min_val'/'max_val' due to floating point
    # inaccuracies or unexpected function behavior.
    normalized_value = max(0.0, min(1.0, normalized_value))

    # --- Color Mapping Strategy (The "ONE Concept" focus) ---
    # WHAT: We are converting a single float (normalized_value) into an RGB color.
    # WHY: This is the crucial step that visualizes the abstract mathematical pattern.
    # We use sine waves for each color channel (Red, Green, Blue) to create a
    # vibrant, cyclic gradient. Each sine wave is offset to produce different
    # hues as the 'normalized_value' changes.

    # Calculate Red component: Offset 0.
    # math.sin(x) ranges from -1 to 1. (math.sin(x) * 0.5 + 0.5) converts it to 0-1.
    red = int((math.sin(normalized_value * math.pi * 2 + 0) * 0.5 + 0.5) * 255)
    # Calculate Green component: Offset by 2/3 pi for a different part of the color cycle.
    green = int((math.sin(normalized_value * math.pi * 2 + math.pi * 2 / 3) * 0.5 + 0.5) * 255)
    # Calculate Blue component: Offset by 4/3 pi for yet another part of the color cycle.
    blue = int((math.sin(normalized_value * math.pi * 2 + math.pi * 4 / 3) * 0.5 + 0.5) * 255)

    # Clamp the final color components to the 0-255 range.
    # WHY: Although the sine wave conversion inherently keeps values within 0-255,
    # this is a good practice for robustness in color calculations.
    red = max(0, min(255, red))
    green = max(0, min(255, green))
    blue = max(0, min(255, blue))

    return (red, green, blue)


def generate_math_art(
    width: int,
    height: int,
    math_function: Callable[[float, float], float],
    min_func_output: float = -1.0,
    max_func_output: float = 1.0,
    output_filename: str = "math_art.png"
) -> Image.Image:
    """
    Generates a pixel art image by applying a mathematical function to each pixel's
    normalized coordinates and mapping the result to a color.

    Args:
        width (int): The width of the generated image in pixels.
        height (int): The height of the generated image in pixels.
        math_function (Callable[[float, float], float]): A function that takes
            two floats (normalized x, y coordinates) and returns a single float.
            This function defines the pattern of the art.
        min_func_output (float): The expected minimum output value from math_function.
            Used for scaling the color mapping.
        max_func_output (float): The expected maximum output value from math_function.
            Used for scaling the color mapping.
        output_filename (str): The name of the file to save the generated image.

    Returns:
        Image.Image: The generated Pillow Image object.
    """
    # WHAT: Create a new blank image using Pillow.
    # WHY: This initializes the canvas where our pixel art will be drawn.
    # 'RGB' mode means Red, Green, Blue color channels. (width, height) are dimensions.
    # (0, 0, 0) sets the initial background color to black.
    image = Image.new('RGB', (width, height), (0, 0, 0))

    # WHAT: Iterate over each pixel in the image.
    # WHY: We need to calculate a color for every single pixel to construct the image.
    # 'y' represents the vertical position, 'x' the horizontal position.
    for y in range(height):
        for x in range(width):
            # WHAT: Normalize pixel coordinates to a common range, e.g., [-1, 1].
            # WHY: This makes mathematical functions easier to apply because they often
            # work well with inputs centered around zero or in a small, consistent range
            # (e.g., sine waves, distances from origin).
            # (x / width) converts x from [0, width-1] to [0, 1].
            # Multiplying by 2 and subtracting 1 shifts it to [-1, 1].
            norm_x = (x / width) * 2 - 1
            norm_y = (y / height) * 2 - 1

            # WHAT: Apply the provided mathematical function to the normalized coordinates.
            # WHY: This is the core logic that defines the unique pattern and complexity of the art.
            try:
                math_result = math_function(norm_x, norm_y)
            except Exception as e:
                # WHAT: Error handling for the math function.
                # WHY: Prevents the program from crashing if a custom math function
                # encounters an invalid input (e.g., division by zero).
                print(f"Error applying math_function at ({x},{y}) with ({norm_x},{norm_y}): {e}")
                math_result = 0.0 # Fallback value to avoid further errors

            # WHAT: Convert the mathematical result into an RGB color.
            # WHY: This visualizes the abstract number generated by the math function
            # as a concrete color on the screen. This uses our `map_value_to_rgb` function.
            pixel_color = map_value_to_rgb(math_result, min_func_output, max_func_output)

            # WHAT: Set the color of the current pixel in the image.
            # WHY: This draws the calculated color onto the image canvas at the specific (x, y) location.
            image.putpixel((x, y), pixel_color)

    # WHAT: Save the generated image to a file.
    # WHY: To persist the generated artwork and view it outside the program.
    image.save(output_filename)
    print(f"Generated '{output_filename}' ({width}x{height} pixels)")

    return image


# --- Example Usage ---
# These examples demonstrate different mathematical functions and their visual outputs.
# Experiment with these and create your own!

if __name__ == "__main__":
    # Example 1: Simple Sine Wave Pattern (Vertical bands)
    # WHAT: A function that uses sine waves along the x-axis.
    # WHY: Demonstrates basic periodic patterns. The output of math.sin is always between -1 and 1.
    def sin_wave_x(x: float, y: float) -> float:
        return math.sin(x * math.pi * 5) # Waves across the x-axis, 5 cycles across the [-1,1] range

    print("\nGenerating 'sin_wave_x.png'...")
    generate_math_art(
        width=400,
        height=300,
        math_function=sin_wave_x,
        min_func_output=-1.0, # Expected min output of sin
        max_func_output=1.0,  # Expected max output of sin
        output_filename="sin_wave_x.png"
    )

    # Example 2: Concentric Circles (Distance from center)
    # WHAT: Uses Euclidean distance from the center (0,0), then applies a sine wave.
    # WHY: Creates concentric ring patterns, showing how geometry translates to art.
    def concentric_circles(x: float, y: float) -> float:
        distance = math.sqrt(x**2 + y**2) # Euclidean distance from the origin (0,0)
        return math.sin(distance * math.pi * 10) # Creates waves based on distance from center

    print("\nGenerating 'concentric_circles.png'...")
    generate_math_art(
        width=400,
        height=400,
        math_function=concentric_circles,
        min_func_output=-1.0, # Expected min output of sin
        max_func_output=1.0,  # Expected max output of sin
        output_filename="concentric_circles.png"
    )

    # Example 3: Checkerboard / Multiplication Pattern
    # WHAT: A function that directly maps the product of x and y coordinates.
    # WHY: Creates hyperbolic-like patterns, demonstrating simple algebraic functions.
    def xy_product(x: float, y: float) -> float:
        return x * y

    print("\nGenerating 'xy_product.png'...")
    generate_math_art(
        width=500,
        height=500,
        math_function=xy_product,
        min_func_output=-1.0, # Min product of x,y in [-1,1] range is -1*1 or 1*-1
        max_func_output=1.0,  # Max product is 1*1 or -1*-1
        output_filename="xy_product.png"
    )

    # Example 4: A more complex fractal-like pattern (using arctan2 and sin)
    # WHAT: Combines `math.atan2` (angle) and `math.sin` with products for intricate patterns.
    # WHY: `math.atan2(y, x)` returns the angle in radians between the positive x-axis and the point (x, y),
    # creating angular patterns. Multiplying by sin(x*y) adds further complexity and distortion.
    def fractal_angle(x: float, y: float) -> float:
        # Shift origin slightly for interesting effect, then multiply for complexity
        # atan2 outputs range from -pi to pi.
        return math.atan2(y * 5, x * 5) * math.sin(x * y * 10)

    print("\nGenerating 'fractal_angle.png'...")
    generate_math_art(
        width=600,
        height=600,
        math_function=fractal_angle,
        # Adjust min/max as the sin multiplier changes the effective range.
        # Max output of atan2 is ~pi. Max sin is 1. Max output is roughly pi * 1.
        # Min output of atan2 is ~-pi. Min sin is -1. Min output is roughly -pi * 1 or pi * -1.
        # Hence, ~+/-pi, which is ~+/-3.14. Using 1.5*pi for robustness.
        min_func_output=-math.pi * 1.5,
        max_func_output=math.pi * 1.5,
        output_filename="fractal_angle.png"
    )

    print("\nAll art generation complete! Check your script directory for the images.")