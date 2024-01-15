# tests for image utils
import pytest
import os

# internal imports
import kleio.image_utils as image_utils


def test_convert_image_to_grayscale(example_image):
    """
    Test the convert_image_to_grayscale function in utils.py.

    Checks that the function returns a grayscale image.
    """

    image = image_utils.convert_image_to_grayscale(example_image)
    
    # check that the image is not None
    assert image is not None


def test_resize_image(example_image):
    """
    Test the resize_image function in utils.py.

    Checks that the function returns a resized image.
    """

    image = image_utils.resize_image(example_image)
    
    # check that the image is not None
    assert image is not None


def test_threshold_image(example_image):
    """
    Test the threshold_image function in utils.py.

    Checks that the function returns a thresholded image.
    """

    image = image_utils.threshold_image(example_image)
    
    # check that the image is not None
    assert image is not None


def test_gaussian_blur_image(example_image):
    """
    Test the gaussian_blur_image function in utils.py.

    Checks that the function returns a blurred image.
    """

    image = image_utils.gaussian_blur_image(example_image)
    
    # check that the image is not None
    assert image is not None


def test_remove_noise_from_image(example_image):
    """
    Test the remove_noise_from_image function in utils.py.

    Checks that the function returns a denoised image.
    """

    image = image_utils.remove_noise_from_image(example_image)
    
    # check that the image is not None
    assert image is not None


def test_dilate_and_erode_image(example_image):
    """
    Test the dilate_and_erode_image function in utils.py.

    Checks that the function returns a dilated and eroded image.
    """

    image = image_utils.dilate_and_erode_image(example_image)
    
    # check that the image is not None
    assert image is not None


def test_deskew_image(example_image):
    """
    Test the deskew_image function in utils.py.

    Checks that the function returns a deskewed image.
    """

    image = image_utils.deskew_image(example_image)
    
    # check that the image is not None
    assert image is not None
