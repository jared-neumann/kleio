# This file provides functions to correct OCR errors in raw text.
# We'll be using langchain to implement LLMs to do the correction.

# third-party imports
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from kleio.general_utils import (
    setup_logger,
)

from kleio.llm_utils import (
    create_openai_prompt,
    create_openai_llm,
    parse_chunks,
)

from kleio.constants import (
    SYS_CORRECTION_MESSAGE,
    HMN_CORRECTION_MESSAGE,
    DEFAULT_CORRECTION_KWARGS,
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

    output = chain.invoke(
        {
            "text": chunk,
        }
    )

    return output


def correct_chunks(text: dict or str, chain, model_name: str, chunk_size: int):
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
    for page in text:
        # get page chunks
        chunks = parse_chunks(
            text=page,
            chunk_size=chunk_size,
            model_name=model_name,
        )

        # append to list of chunks
        chunks_by_page.append(chunks)

    corrected_pages = []

    # iterate through the chunks per page
    # and correct each one
    try:
        for page in tqdm(chunks_by_page):
            corrected_page = []
            for chunk in page:
                corrected_chunk = correct_chunk(chunk, chain)
                corrected_page.append(corrected_chunk)
            corrected_pages.append("".join(corrected_page))
    except Exception as e:
        logger.error(f"Error while correcting chunks: {e}")
        return None

    return corrected_pages


# we need to create a router function
def get_correction(
    text: dict or str,
    api_key: str,  # api key has to match llm provider TODO: add more providers
    llm_provider: str = "openai",  # one of ['openai', 'huggingface', etc.]
    model_name: str = "gpt-3.5-turbo-16k",  # also has to match accordingly
    temperature: int = 0,
    chunk_size: int = 1024,
    system_message: str = SYS_CORRECTION_MESSAGE,
    human_message: str = HMN_CORRECTION_MESSAGE,
    more_info: dict = DEFAULT_CORRECTION_KWARGS,
):
    """
    Get a correction from an LLM.
    """
    logger.info("Getting correction from LLM")

    # CASE 1: openai
    if llm_provider == "openai":
        # create the prompt
        prompt = create_openai_prompt(
            system_message=system_message,
            human_message=human_message,
            more_info=more_info,
        )

        # create the llm
        llm = create_openai_llm(
            api_key=api_key, model_name=model_name, temperature=temperature
        )
    else:
        logger.error("LLM provider currently supported")
        return None

    # create the output parser
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | llm | output_parser

    corrected_pages = correct_chunks(
        chain=chain, text=text, model_name=model_name, chunk_size=chunk_size
    )

    return corrected_pages
