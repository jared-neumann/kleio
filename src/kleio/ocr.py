"""
ocr.py

This file provides functions to process different types of files
(images, PDFs, and text files) and extract text from them using OCR
(Optical Character Recognition) if necessary. By default, is treated
by the page, and there is a general function to pass any filepath
or string, and return a dictionary of the filename, extension, and
pages of text.
"""
# built-in imports
import os

# third-party imports
import pytesseract
import fitz  # PyMuPDF
import pdf2image
import numpy as np
import cv2

# self imports
from kleio.general_utils import (
    get_file_extension,
    pdf_has_text,
    format_text_string,
)

from kleio.image_utils import (
    setup_logger,
    convert_image_to_grayscale,
    resize_image,
    threshold_image,
    deskew_image,
    dilate_and_erode_image,
    gaussian_blur_image,
)

# set up logging
logger = setup_logger(__name__)

IMAGE_CONFIG = {
    "grayscale": True,
    "resize": False,
    "threshold": True,
    "deskew": False,
    "dilate_and_erode": False,
    "blur": False,
}


# for image options, we need to preprocess given the functions in utils.py
# and a given config of what to do to the image
def preprocess_image(image: np.ndarray, image_kwargs: dict = None) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.

    Args:
        image (np.ndarray): Image to preprocess.
        grayscale (bool): Whether to convert the image to grayscale.
        resize (bool): Whether to resize the image.
        threshold (bool): Whether to threshold the image.
        deskew (bool): Whether to deskew the image.
        dilate_and_erode (bool): Whether to dilate and erode the image.
        blur (bool): Whether to blur the image.

    Returns:
        np.ndarray: Preprocessed image.
    """
    logger.info("Preprocessing image")

    # convert to grayscale
    if image_kwargs["grayscale"]:
        image = convert_image_to_grayscale(image)
    # resize image
    if image_kwargs["resize"]:
        image = resize_image(image)
    # threshold image
    if image_kwargs["threshold"]:
        image = threshold_image(image)
    # deskew image
    if image_kwargs["deskew"]:
        image = deskew_image(image)
    # dilate and erode image
    if image_kwargs["dilate_and_erode"]:
        image = dilate_and_erode_image(image)
    # blur image
    if image_kwargs["blur"]:
        image = gaussian_blur_image(image)

    return image


# 1. image file
def get_text_from_image(filepath, image_kwargs: dict = None):
    """
    Get text from an image file.

    Args:
        filepath (str): Path to the image file to extract text from.

    Returns:
        str: Extracted text from the image file.
    """
    logger.info(f"Getting page text from image file {filepath}")

    # load image
    try:
        image = cv2.imread(filepath)
    except Exception as e:
        logger.error(f"Error while loading image file {filepath}: {e}")
        return None

    # preprocess image
    try:
        image = preprocess_image(image, image_kwargs)
    except Exception as e:
        logger.error(f"Error while preprocessing image file {filepath}: {e}")
        return None

    page_text = ""
    try:
        page_text = pytesseract.image_to_string(filepath)
        page_text = format_text_string(page_text)
    except Exception as e:
        logger.error(f"Error while extracting text from image file {filepath}: {e}")

    return page_text


# 2. directory of image files
def get_text_from_image_directory(dirpath, image_kwargs: dict = None):
    """
    Get text from a directory of image files.

    Args:
        dirpath (str): Path to the directory of image files to extract text from.

    Returns:
        list: Extracted page text from the directory of image files.
    """
    logger.info(f"Getting page text from image directory {dirpath}")

    # get a list of all image files in the directory
    try:
        file_list = os.listdir(dirpath)
        filenames = [os.path.join(dirpath, file) for file in file_list]
        image_files = [
            file
            for file in filenames
            if get_file_extension(file) in ["JPG", "JPEG", "PNG"]
        ]
    except Exception as e:
        logger.error(f"Error while getting image files from directory {dirpath}: {e}")
        return None

    # get text from each image file
    page_texts = []
    for image_file in image_files:
        page_text = get_text_from_image(image_file, image_kwargs)
        page_texts.append(page_text)

    return page_texts


# 3. PDF file with text
def get_text_from_pdf_with_text(filepath):
    """
    Get text from a PDF file that already has text.

    Args:
        filepath (str): Path to the PDF file to extract text from.

    Returns:
        list: Extracted page text from the PDF file.
    """
    logger.info(f"Getting text from PDF file {filepath}")
    page_texts = []
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                page_text = page.get_text()
                page_text = format_text_string(page_text)
                page_texts.append(page_text)
    except Exception as e:
        logger.error(f"Error while extracting text from PDF file {filepath}: {e}")
    return page_texts


# 4. PDF file with without text
def get_text_from_pdf_without_text(filepath, image_kwargs: dict = None):
    """
    Get text from a PDF file that does not have text.

    Args:
        filepath (str): Path to the PDF file to extract text from.

    Returns:
        list: Extracted page text from the PDF file.
    """
    logger.info(f"Getting text from PDF file {filepath}")
    page_texts = []
    try:
        # convert PDF to image
        logger.info(f"Converting PDF file {filepath} to image")
        images = pdf2image.convert_from_path(filepath)

        # get text from each image
        for image in images:
            image = np.array(image)
            page_text = pytesseract.image_to_string(image)
            page_text = format_text_string(page_text)
            page_texts.append(page_text)

    except Exception as e:
        logger.error(f"Error while extracting text from PDF file {filepath}: {e}")

    return page_texts


# 5. text file
def get_text_from_text_file(filepath):
    """
    Get text from a text file.

    Args:
        filepath (str): Path to the text file to extract text from.

    Returns:
        str: Extracted text (will be treated as a single page) from
            the text file.
    """
    logger.info(f"Getting text from text file {filepath}")
    page_text = ""
    try:
        with open(filepath, "r") as f:
            page_text = f.read()
        page_text = format_text_string(page_text)
    except Exception as e:
        logger.error(f"Error while extracting text from text file {filepath}: {e}")
    return page_text


# 6. text string
def get_text_from_text_string(text):
    """
    Get text from a text string.

    Args:
        text (str): Text string to extract text from.

    Returns:
        str: Extracted text (will be treated as a single page) from the text string.
    """
    # For raw text, we want to ensure proper encoding
    # and remove any non-printable characters
    logger.info("Getting text from text string")
    try:
        page_text = format_text_string(text)
    except Exception as e:
        logger.error(f"Error decoding text string: {e}")
    return page_text
    ...


# main function
# routes to the appropriate function based on the input
def retrieve_text(filepath_dir_or_string: str, image_kwargs: dict = IMAGE_CONFIG):
    """
    Get text from a file or text string.

    Args:
        filepath_or_text (str): Path to a file or a text string.

    Returns:
        dict: {'filename': filename, 'extension': extension, 'pages': [page_text, ...]}
    """
    logger.info(f"Retrieving text from {filepath_dir_or_string}")

    # CASE 0: text string
    if (
        isinstance(filepath_dir_or_string, str)
        and not os.path.isfile(filepath_dir_or_string)
        and not os.path.isdir(filepath_dir_or_string)
    ):
        logger.info("String provided")
        return {
            "filename": None,
            "extension": None,
            "pages": [get_text_from_text_string(filepath_dir_or_string)],
        }

    # otherwise, check if the input is a file
    elif os.path.isfile(filepath_dir_or_string):
        logger.info("File provided")

        extension = get_file_extension(filepath_dir_or_string)

        # CASE 1: PDF file
        if extension == "PDF":
            # CASE 1a: PDF file with text
            if pdf_has_text(filepath_dir_or_string):
                return {
                    "filename": os.path.basename(filepath_dir_or_string),
                    "extension": extension,
                    "pages": get_text_from_pdf_with_text(filepath_dir_or_string),
                }

            # CASE 1b: PDF file without text
            else:
                return {
                    "filename": os.path.basename(filepath_dir_or_string),
                    "extension": extension,
                    "pages": get_text_from_pdf_without_text(
                        filepath_dir_or_string, image_kwargs
                    ),
                }

        # CASE 2: image file
        elif extension in ["JPG", "JPEG", "PNG"]:
            return {
                "filename": os.path.basename(filepath_dir_or_string),
                "extension": extension,
                "pages": [get_text_from_image(filepath_dir_or_string, image_kwargs)],
            }

        # CASE 3: text file
        elif extension == "TXT":
            return {
                "filename": os.path.basename(filepath_dir_or_string),
                "extension": extension,
                "pages": [get_text_from_text_file(filepath_dir_or_string)],
            }

        else:
            logger.error(f"File extension {extension} not supported")
            return None

    # CASE 4: directory of image files
    elif os.path.isdir(filepath_dir_or_string):
        logger.info("Directory provided")
        return {
            "filename": None,
            "extension": None,
            "pages": get_text_from_image_directory(
                filepath_dir_or_string, image_kwargs
            ),
        }

    else:
        logger.error("Input not supported")
        return None
