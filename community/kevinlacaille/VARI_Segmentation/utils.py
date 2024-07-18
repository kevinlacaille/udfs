import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import exiftool
import json


def get_bands(image_path):
    """
    Open a geospatial image and read the blue, green, and red bands.

    Parameters:
    -----------
    image_path : str
        The file path to the geospatial image.

    Returns:
    --------
    blue : numpy.ndarray
        The blue band of the image as a NumPy array.
    green : numpy.ndarray
        The green band of the image as a NumPy array.
    red : numpy.ndarray
        The red band of the image as a NumPy array.
    rgb : numpy.ndarray
        A 3-dimensional array combining the blue, green, and red bands into an RGB image.
    """

    # Open the image and read the bands as numpy arrays
    with rasterio.open(image_path) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)

    rgb = np.dstack((blue, green, red))

    return blue, green, red, rgb


def get_metadata(image_path):
    """
    Extract metadata from a geospatial image.

    Parameters:
    -----------
    image_path : str
        The file path to the geospatial image.

    Returns:
    --------
    metadata : dict
        A dictionary containing the metadata of the image.
    """

    # Get the metadata of the image
    with exiftool.ExifTool() as et:
        metadata = json.loads(et.execute(b'-j', image_path))

    return metadata


def get_gsd(metadata, height=15):
    """
    Calculate the Ground Sampling Distance (GSD) for a given image,
    both at the altitude and at a specified height above ground.

    Parameters:
    -----------
    metadata : dict
        A dictionary containing the metadata of the image.
    height : float, optional
        The height above ground for which to calculate the GSD, in meters. Default is 15 meters.

    Returns:
    --------
    gsd : float
        The GSD at the altitude of the image capture, in meters per pixel.
    gsd_tree : float
        The GSD at the specified height above ground, in meters per pixel.
    """

    # Extract the GPS Altitude
    altitude = float(metadata[0].get("XMP:RelativeAltitude"))
    print(f"Altitude: {altitude} m")

    # Extract the focal length of the camera
    focal_length = metadata[0].get("EXIF:FocalLength")  # in mm
    # Extract the image's width
    image_width = metadata[0].get("File:ImageWidth")

    # Compute the pixel pitch
    pixel_pitch = 6.17e-3 / image_width

    # Compute the global GSD and the GSD at the tree top
    gsd = (altitude) * pixel_pitch / (focal_length / 1000)
    gsd_tree = (altitude - height) * pixel_pitch / (focal_length / 1000)

    return gsd, gsd_tree


def get_vari(blue, green, red):
    """
    Calculate the Visible Atmospherically Resistant Index (VARI)
    using the blue, green, and red bands of an image.

    Parameters:
    -----------
    blue : numpy.ndarray
        The blue band of the image as a NumPy array.
    green : numpy.ndarray
        The green band of the image as a NumPy array.
    red : numpy.ndarray
        The red band of the image as a NumPy array.

    Returns:
    --------
    vari : numpy.ndarray
        The calculated VARI index as a NumPy array.
    """

    # Calculate the VARI index
    vari = (green.astype(float) - red.astype(float)) / (
        green.astype(float) + red.astype(float) - blue.astype(float))

    return vari


def threshold(vari, vari_min=0.1):
    """
    Generate vegetation and non-vegetation masks based on VARI index thresholds.

    Parameters:
    -----------
    vari : numpy.ndarray
        The VARI index as a NumPy array.
    vari_min : float, optional
        The minimum threshold for the VARI index to classify vegetation. Default is 0.1.

    Returns:
    --------
    vegetation_mask : numpy.ndarray
        A mask where vegetation areas are marked with 1 and non-vegetation areas are NaN.
    non_vegetation_mask : numpy.ndarray
        A mask where non-vegetation areas are marked with 1 and vegetation areas are NaN.
    """

    # Generate the vegetation mask
    vegetation_mask = np.full(vari.shape, np.nan)
    vegetation_mask[(vari >= vari_min)] = 1

    # Generate the non-vegetation mask
    non_vegetation_mask = np.full(vari.shape, np.nan)
    non_vegetation_mask[vari < vari_min] = 1

    return vegetation_mask, non_vegetation_mask


def smoothing(mask, kernel_size=7):
    """
  Apply Gaussian blur to smooth the edges of a mask.

  Parameters:
  -----------
  mask : numpy.ndarray
      The binary mask to be smoothed.
  kernel_size : int, optional
      The size of the kernel to be used in the Gaussian blur. Default is 7.

  Returns:
  --------
  blur : numpy.ndarray
      The smoothed mask after applying Gaussian blur.
  """

    # Apply a Gaussian blur to the mask
    blur = cv.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    return blur


def morphological_operations(mask, kernel_size=19):
    """
    Apply opening and closing morphological operations to a mask.

    Parameters:
    -----------
    mask : numpy.ndarray
        The binary mask to be processed.
    kernel_size : int, optional
        The size of the kernel to be used in the morphological operations. Default is 19.

    Returns:
    --------
    closing : numpy.ndarray
        The mask after applying opening followed by closing morphological operations.
    """

    # Define the kernel size
    opening_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply the morphological filters
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, opening_kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, closing_kernel)

    return closing


def segmentation(mask, rgb):
    """
    Apply a mask to the RGB image to create a segmentation overlay and its inverse.

    Parameters:
    -----------
    mask : numpy.ndarray
        The binary mask to be applied to the RGB image.
    rgb : numpy.ndarray
        The original RGB image.

    Returns:
    --------
    mask_overlay : numpy.ndarray
        The RGB image with the mask overlay applied, highlighting the masked areas in purple.
    inverse_mask_overlay : numpy.ndarray
        The RGB image with the inverse mask overlay applied, highlighting the non-masked areas in purple.
    """

    # Apply the mask to the RGB image
    mask_overlay = np.zeros_like(rgb)
    mask_overlay[:, :, 0] = 128  # Red channel for purple
    mask_overlay[:, :, 1] = 0  # Green channel for purple
    mask_overlay[:, :, 2] = 128  # Blue channel for purple

    # Apply the filtered mask to the mask overlay
    mask_overlay[mask != 1] = [0, 0, 0]

    # Create the inverse mask
    inverse_mask = np.logical_not(mask).astype(np.uint8)

    # Create an inverse mask overlay with purple color
    inverse_mask_overlay = np.zeros_like(rgb)
    inverse_mask_overlay[:, :, 0] = 128  # Red channel for purple
    inverse_mask_overlay[:, :, 1] = 0  # Green channel for purple
    inverse_mask_overlay[:, :, 2] = 128  # Blue channel for purple

    # Apply the inverse mask to the inverse mask overlay
    inverse_mask_overlay[mask == 1] = [0, 0, 0]

    return mask_overlay, inverse_mask_overlay
