# conftest.py is a file that pytest automatically looks for in the tests directory.
# It contains fixtures and other configuration information for pytest.
# Let's define some data that we can use to test our functions.
import pytest
import os

# examples for testing get_file_extension
@pytest.fixture()
def example_filenames_with_extensions():
    return [
        ("example.txt", "TXT"),
        ("example.pdf", "PDF"),
        ("example.png", "PNG"),
        ("example.jpg", "JPG"),
        ("example.jpeg", "JPEG"),
        ("example.gif", "GIF"),
        ("example.doc", "DOC"),
        ("example.docx", "DOCX"),
        ("example.ppt", "PPT"),
        ("example.pptx", "PPTX"),
        ("example.xls", "XLS"),
        ("example.xlsx", "XLSX"),
        ("example.csv", "CSV"),
        ("example.zip", "ZIP"),
        ("example.tar.gz", "GZ"),
        ("example", None),
        ("example.", None),
        ("", None),
        (None, None),
    ]

@pytest.fixture()
def example_data_filepaths():
    
    # relative path to directory
    directory = "tests/test_input"

    # iterate over files in directory
    filepaths = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        filepaths.append(filepath)

    return filepaths

