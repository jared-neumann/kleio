# image utility functions
# third-party modules
import numpy as np
import cv2

# internal modules
from kleio.general_utils import setup_logger

# set up logging
logger = setup_logger(__name__)


def convert_image_to_grayscale(image):
    """
    Convert a color image to grayscale.

    Args:
        image (numpy.ndarray): Color image to be converted.

    Returns:
        numpy.ndarray: Grayscale version of the image.
    """
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.warn(f"Error converting image to grayscale, proceeding without it: {e}")
    return image


def resize_image(image, dpi=300):
    """
    Resize an image to a specified DPI.

    Args:
        image (numpy.ndarray): Image to be resized.
        dpi (int): Target DPI for resizing.

    Returns:
        numpy.ndarray: Resized image.
    """
    height, width = image.shape[:2]
    if height * width == 0:
        logger.warning("Either height or width or both is 0, returning original image")
        return image
    elif dpi == 0:
        logger.warning("DPI is set to 0, returning original image")
        return image
    else:
        try:
            new_height, new_width = int(height * dpi / 72), int(width * dpi / 72)
            image = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
        except Exception as e:
            logger.warning(f"Error while resizing image, proceeding without it: {e}")
    return image


def threshold_image(image):
    """
    Apply binary thresholding to an image using Otsu's method.

    Args:
        image (numpy.ndarray): Image to be thresholded.

    Returns:
        numpy.ndarray: Thresholded (binary) image.
    """
    try:
        image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except Exception as e:
        logger.warning(f"Error while thresholding image, proceeding without it: {e}")
    return image


def deskew_image(image, limit=5):
    """
    Deskew an image based on its content.

    Args:
        image (numpy.ndarray): Image to be deskewed, assumed to be preprocessed.
        limit (float): Minimum skew angle required to perform deskewing.

    Returns:
        numpy.ndarray: Deskewed image, or original image if below skew limit.
    """
    try:
        # determine the angle of the image
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle

        # if the angle is too large, deskew the image
        if abs(angle) >= limit:
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            image = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

    except Exception as e:
        logger.warning(f"Returning original image. Error while deskewing image: {e}")

    return image


def dilate_and_erode_image(image, kernel_size=1, iterations=1):
    """
    Apply dilation followed by erosion to an image.

    Args:
        image (numpy.ndarray): Image to be processed.
        kernel_size (int): Size of the kernel used for dilation and erosion.
        iterations (int): Number of times dilation and erosion are applied.

    Returns:
        numpy.ndarray: Image after dilation and erosion.
    """
    adjusted_image = None
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        adjusted_image = cv2.erode(
            cv2.dilate(image, kernel, iterations=iterations),
            kernel,
            iterations=iterations,
        )
    except Exception as e:
        logger.warning(
            f"Returning original image. Error while dilating and eroding image: {e}"
        )
        adjusted_image = image
    return adjusted_image


def remove_noise_from_image(image, kernel_size=5):
    """
    Remove noise from an image using median blurring.

    Args:
        image (numpy.ndarray): Image from which noise is to be removed.
        kernel_size (int): Size of the kernel used for median blur.

    Returns:
        numpy.ndarray: Image with reduced noise.
    """
    try:
        image = cv2.medianBlur(image, kernel_size)
    except Exception as e:
        logger.warning(f"Returning original image. Error while removing noise: {e}")
    return image


def gaussian_blur_image(image, kernel_size=5):
    """
    Apply Gaussian blur to an image.

    Args:
        image (numpy.ndarray): Image to be blurred.
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        numpy.ndarray: Blurred image.
    """
    try:
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    except Exception as e:
        logger.warning(f"Returning original image. Error while blurring image: {e}")
    return image
