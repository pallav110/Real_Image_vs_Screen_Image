"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import sys
from PIL import Image


def predict(image_path: str) -> float:
    img = Image.open(image_path).convert("RGB")
    # TODO: run your detector and return how likely the image is a photo-of-a-screen.
    # It can be a trained model, a classic CV / image-processing method, frequency
    # analysis, or any algorithm you like -- your choice.
    raise NotImplementedError("return a fraud score in [0, 1]")


if __name__ == "__main__":
    print(predict(sys.argv[1]))
