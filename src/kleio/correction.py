# This file provides functions to correct OCR errors in raw text.
# We'll be using langchain to implement LLMs to do the correction.
import os

# third-party imports
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from kleio.general_utils import (
    setup_logger,
)

from kleio.llm_utils import (
    get_tokenizer,
    create_prompt,
    create_llm,
    parse_chunks,
)

from kleio.constants import (
    SYS_CORRECTION_MESSAGE,
    HMN_CORRECTION_MESSAGE
)

# set up logging
logger = setup_logger(__name__)


# we need a function to iterate through pages
# chunk them appropriately
# and get corrections from the llm
# reassembling page text as we go
def correct_chunk(chunk: str, chain):
    """
    Correct a chunk of text.
    """

    output = chain.invoke({"text": chunk})

    return output


def correct_chunks(text: dict or str, chain, model_name: str, chunk_size: int, output_path: str):
    """
    Correct chunks of text.

    Args:
        text (dict or str): Text to be corrected. Can be a string or a dictionary
            containing a list of pages.
        chain: Langchain chain to be used for correction.
        model_name (str): Name of the model to be used for correction.
        chunk_size (int): Size of the chunks to be used for correction.

    Returns:
        list: List of corrected pages.
    """

    # first, we need to check if the text is a string or a dictionary
    # if it's a string, we need to wrap it in a list and chunk it as is
    # if it's a dictionary, we need to chunk each page separately
    # and then reassemble either as pages in a dictionary
    if isinstance(text, str):
        text = [text]
    elif isinstance(text, dict):
        if "pages" in text:
            text = text["pages"]
        else:
            logger.error("Text dictionary does not contain a 'pages' key")
            return None
    else:
        logger.error("Text is neither a string nor a dictionary")
        return None

    # create a list for chunks
    chunks_by_page = []
    logger.info("Parsing chunks...")

    # create a tokenizer
    tokenizer = None
    try:
        tokenizer = get_tokenizer(model_name)
    except Exception as e:
        logger.error(f"Error while getting tokenizer: {e}")

    for page in tqdm(text):
        # get page chunks
        chunks = parse_chunks(
            text=page,
            chunk_size=chunk_size,
            model_name=model_name,
            tokenizer=tokenizer,
        )

        # append to list of chunks
        chunks_by_page.append(chunks)

    corrected_pages = []

    # create an output file if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write("")
    else: # erase the contents of the file if it exists
        with open(output_path, "r+") as f:
            f.truncate(0)

    # iterate through the chunks per page
    # and correct each one
    try:
        for page in tqdm(chunks_by_page):
            index = chunks_by_page.index(page)
            logger.info("Correcting chunks for page %s", index)

            corrected_page = []
            for chunk in page:
                corrected_chunk = correct_chunk(chunk, chain)
                corrected_page.append(corrected_chunk)
            corrected_page = "".join(corrected_page)

            # write to file
            with open(output_path, "a") as f:
                f.write(corrected_page)

            # append to list of corrected pages
            corrected_pages.append("".join(corrected_page))
    except Exception as e:
        logger.error(f"Error while correcting chunks: {e}")
        return None

    return corrected_pages


# we need to create a router function
def get_correction(
    text: dict or str,
    api_key: str, # "not-needed" for local models
    model_name: str = "gpt-3.5-turbo-16k",
    base_url: str = None,
    output_path: str = "correction.txt",
    temperature: int = 0,
    chunk_size: int = 2048,
    system_message: str = SYS_CORRECTION_MESSAGE,
    human_message: str = HMN_CORRECTION_MESSAGE,
    filename: str="N/A",
    filetype: str="N/A",
    ocr_software: str="N/A",
    image_preprocessing_software: str="N/A",
    date: str="N/A",
    language: str="N/A",
    comments: str="N/A",
):
    """
    Get a correction from an LLM.
    """
    logger.info("Getting correction from LLM")

    # create the more info dictionary
    more_info = {
        "filename": filename,
        "filetype": filetype,
        "ocr_software": ocr_software,
        "image_preprocessing_software": image_preprocessing_software,
        "date": date,
        "language": language,
        "comments": comments,
    }

    # create the prompt
    prompt = None
    try:
        prompt = create_prompt(
            system_message=system_message,
            human_message=human_message,
            more_info=more_info,
        )
    except Exception as e:
        logger.error(f"Error while creating prompt: {e}")

    # create the llm
    llm = None
    try:
        llm = create_llm(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
        )
    except Exception as e:
        logger.error(f"Error while creating LLM: {e}")

    # create the output parser
    output_parser = None
    try:
        output_parser = StrOutputParser()
    except Exception as e:
        logger.error(f"Error while creating output parser: {e}")

    # create the chain
    chain = None
    if prompt and llm and output_parser:
        chain = prompt | llm | output_parser
    else:
        logger.error("Could not create chain")

    # correct the text
    corrected_pages = None
    if chain:
        try:
            corrected_pages = correct_chunks(
                text=text,
                chain=chain,
                model_name=model_name,
                chunk_size=chunk_size,
                output_path=output_path,
            )
        except Exception as e:
            logger.error(f"Error while correcting text: {e}")

    return corrected_pages
