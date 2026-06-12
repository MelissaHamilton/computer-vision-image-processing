"""
Computer Vision -- image filtering showcase.

Loads one image and applies a range of classic OpenCV filters, then writes them
into a single labelled grid (comparison.png) so the differences are obvious at a
glance -- e.g. on a GitHub README.

Grew out of a Colab notebook that did just blur + sharpen; this adds grayscale,
edge detection and thresholding to show more of the toolbox.

Pure OpenCV + NumPy. No Google Drive / Colab / matplotlib needed.
"""

import cv2
import numpy as np

image_path = "sc.jpg"  # was a Google Drive path in Colab; now just the local file

# --- Load (imread returns None on a bad path, so check explicitly) --------
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error: Image not loaded. Check the file path.")
print("Image loaded successfully.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- The filters ----------------------------------------------------------
# 1. Gaussian blur: average each pixel with neighbours -> smooth / less detail
blurred = cv2.GaussianBlur(image, (15, 15), 0)

# 2. Sharpen: hand-written kernel amplifying a pixel vs. its neighbours
sharpen_kernel = np.array([[0, -1, 0],
                           [1,  5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

# 3. Canny edge detection: find intensity boundaries
edges = cv2.Canny(image, 100, 200)

# 4. Binary threshold (on the grayscale): every pixel -> black or white
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Single-channel results -> back to 3 channels so they stack with color images
gray3   = cv2.cvtColor(gray,   cv2.COLOR_GRAY2BGR)
edges3  = cv2.cvtColor(edges,  cv2.COLOR_GRAY2BGR)
thresh3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# --- Lay them out in a labelled grid --------------------------------------
def label(img, text):
    """Add a caption bar above the image."""
    bar = np.zeros((40, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])

panels = [
    label(image,     "1. Original"),
    label(gray3,     "2. Grayscale"),
    label(blurred,   "3. Gaussian Blur"),
    label(sharpened, "4. Sharpen"),
    label(edges3,    "5. Canny Edges"),
    label(thresh3,   "6. Threshold"),
]

def row(items):
    gap = np.full((items[0].shape[0], 8, 3), 255, dtype=np.uint8)
    out = items[0]
    for p in items[1:]:
        out = np.hstack([out, gap, p])
    return out

top    = row(panels[0:3])
bottom = row(panels[3:6])
vgap   = np.full((8, top.shape[1], 3), 255, dtype=np.uint8)
grid   = np.vstack([top, vgap, bottom])

cv2.imwrite("comparison.png", grid)
print("Saved comparison.png  (size:", grid.shape, ")")
