# created under assist of Claude, it converts handwritten images to fit the size and invert to white and black
from PIL import Image
import numpy as np

def prepare_digit(input_path, output_path):
    # Load and convert to grayscale
    img = Image.open(input_path).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Convert to array
    arr = np.array(img)
    
    # Check if we need to invert (MNIST has white on black)
    if arr.mean() > 127:  # Black digit on white background
        arr = 255 - arr  # Invert to white on black
    
    # Save
    Image.fromarray(arr).save(output_path)
    print(f"Saved {output_path}")

# Process all 10 digits
for i in range(10):
    prepare_digit(f'createdDigits/{i}.png', f'createdDigits/{i}.png')