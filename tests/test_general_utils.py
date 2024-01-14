# generate pytests for utils.py
# built-in modules
import pytest
import os

# internal imports
import kleio.general_utils as general_utils


def test_setup_logger():
    """
    Test the setup_logger function in utils.py.

    Checks that the function returns a logger object.
    """

    logger = general_utils.setup_logger(__name__)
    assert logger is not None


def test_get_file_extension(example_filenames_with_extensions):
    """
    Test the get_file_extension function in utils.py.

    Checks that the function returns the correct file 
    extension for a given example filename.
    """

    for filename, expected_extension in example_filenames_with_extensions:
        extension = general_utils.get_file_extension(filename)
        assert extension == expected_extension


def test_pdf_has_text(example_data_filepaths):
    """
    Test the pdf_has_text function in utils.py.

    Checks that the function returns True for PDF files
    that have text, and False for PDF files that do not
    have text.
    """

    for filepath in example_data_filepaths:
        if 'test_0' in filepath:
            assert general_utils.pdf_has_text(filepath) == True
        if 'test_1' in filepath:
            assert general_utils.pdf_has_text(filepath) == False
        if 'test_2' in filepath:
            assert general_utils.pdf_has_text(filepath) == False
        else:
            continue


def test_format_text_string():
    """
    Test the format_text_string function in utils.py.

    Checks that the function returns a string.
    """

    text = general_utils.format_text_string("x√ab c")

    # check for utf-8 encoding
    assert text == "x√ab c"
    assert isinstance(text, str)
