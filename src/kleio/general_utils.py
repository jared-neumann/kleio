# general utility functions
# built-in imports
import logging
import sys
import os

# third-party imports
import fitz  # PyMuPDF


def setup_logger(__name__):
    """
    Set up logging for the module.

    Args:
        __name__ (str): Name of the module to log for.

    Returns:
        logging.Logger: Configured logger instance for the module.
    """
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger(__name__)


def get_file_extension(filename):
    """
    Determine the file extension of a given filename.

    Args:
        filename (str): Filename to extract the extension from.

    Returns:
        str: The extracted file extension in uppercase, or None if not present.
    """
    extension = None
    if filename is not None:
        extension = os.path.splitext(filename)[1][1:].upper()
        extension = None if extension == "" else extension
    else:
        logger.info("No filename provided")
    return extension


def pdf_has_text(filepath):
    """
    Check whether a PDF file contains text.

    Args:
        filepath (str): Path to the PDF file to be checked.

    Returns:
        bool: True if the PDF contains text, False otherwise.
    """
    has_text = False
    if get_file_extension(filepath) != "PDF":
        logger.info(f"File {filepath} is not a PDF file")
        has_text = False

    try:
        with fitz.open(filepath) as doc:
            # sample up to ten pages
            text = "".join([page.get_text() for page in doc.pages(0, 10)])
            if text and len(text) > 0:
                has_text = True
    except Exception as e:
        logger.warning(f"Error while loading PDF file {filepath}: {e}")
        has_text = False
    return has_text


def format_text_string(text):
    """
    Forces encoding of text to UTF-8 and removes non-printable characters.

    Args:
        text (str): Text string to be formatted.

    Returns:
        str: Formatted text string.
    """
    try:
        text = text.encode("utf-8", "ignore").decode()
    except Exception as e:
        logger.error(f"Error decoding text string: {e}")
    return text
