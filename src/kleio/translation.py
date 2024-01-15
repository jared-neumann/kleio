# TODO: Implement translation
"""
This file provides functions for translating text.
The input is expected to be either a dict or string.
Text is retrieved.
Then it is parsed into sentences.
The sentences are chunked.
The chunks are translated.
The translated chunks are concatenated.
The concatenated chunks are returned.
"""
# third-party imports
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from kleio.general_utils import (
    setup_logger,
)

from kleio.llm_utils import (
    create_openai_prompt,
    create_openai_llm,
    parse_sentence_chunks,
)

from kleio.constants import (
    SYS_TRANSLATION_MESSAGE,
    HMN_TRANSLATION_MESSAGE,
    DEFAULT_TRANSLATION_KWARGS,
)

# set up logging
logger = setup_logger(__name__)


def translate_chunk(chunk: str, chain):
    """
    Translate a chunk of text.
    """

    output = chain.invoke(
        {
            "text": chunk,
        }
    )

    return output


def translate_chunks(
    text: str, chain, model_name: str = "gpt-3.5-turbo", chunk_size: int = 1024
):
    """
    Translate chunks of text.

    Args:
        text (dict or str): Text to be translated. Can be a string or a dictionary
            containing a list of pages.
        chain: Langchain chain to be used for translation.
        model_name (str): Name of the model to be used for translation.
        chunk_size (int): Size of the chunks to be used for translation.

    Returns:
        list: List of translated pages.
    """

    # first, we need to retrieve the text
    if isinstance(text, dict):
        text = text["text"]

        # then, we need to concatenate the pages
        text = " ".join(text)

    # then, we need to parse the text into chunked sentences
    chunks = parse_sentence_chunks(text, model_name, chunk_size)

    # then, we need to translate the chunks
    translated_chunks = []

    for chunk in tqdm(chunks):
        translated_chunk = translate_chunk(chunk, chain)
        translated_chunks.append(translated_chunk)

    return translated_chunks


def translate(
    text: dict or str,
    api_key: str,
    model_name: str,
    llm_provider: str,
    temperature: int = 0,
    chunk_size: int = 1024,
    system_message: str = SYS_TRANSLATION_MESSAGE,
    human_message: str = HMN_TRANSLATION_MESSAGE,
    more_info: dict = DEFAULT_TRANSLATION_KWARGS,
):
    """
    Translate text.

    Args:
        text (dict or str): Text to be translated. Can be a string or a dictionary
            containing a list of pages.
        api_key (str): API key for the LLM provider.
        model_name (str): Name of the model to be used for translation.
        llm_provider (str): Name of the LLM provider.
        temperature (int): Temperature to be used for translation.
        chunk_size (int): Size of the chunks to be used for translation.
        system_message (str): System message to be used for translation.
        human_message (str): Human message to be used for translation.
        more_info (dict): Dictionary containing additional information to be used
            for translation.

    Returns:
        str: String of the translated text.
    """

    if isinstance(text, dict):
        if "pages" in text:
            text = text["pages"]
            text = " ".join(text)  # TODO: figure out how to handle pages
        else:
            logger.error("Text dictionary does not contain a 'pages' key")
            return None
    elif isinstance(text, str):
        pass
    else:
        logger.error("Text is neither a string nor a dictionary")
        return None

    if llm_provider == "openai":
        # create the prompt
        prompt = create_openai_prompt(
            system_message=system_message,
            human_message=human_message,
            more_info=more_info,
        )

        # create the llm
        llm = create_openai_llm(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )

    # output parser
    output_parser = StrOutputParser()

    # create the chain
    chain = prompt | llm | output_parser

    # translate the chunks
    translated_chunks = translate_chunks(text, chain, model_name, chunk_size)

    # concatenate the chunks
    translation = " ".join(translated_chunks)

    return translation
