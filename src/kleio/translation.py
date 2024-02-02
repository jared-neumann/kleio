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
    parse_sentence_chunks,
)

from kleio.constants import (
    SYS_TRANSLATION_MESSAGE,
    HMN_TRANSLATION_MESSAGE
)

# set up logging
logger = setup_logger(__name__)


def translate_chunk(chunk: str, chain):
    """
    Translate a chunk of text.
    """

    output = chain.invoke({"text": chunk})

    return output


def translate_chunks(
    text: str, chain, model_name: str, chunk_size: int, output_path: str, tokenizer
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
    logger.info("Parsing text into chunks...")
    chunks = parse_sentence_chunks(text, model_name, chunk_size, tokenizer)
    logger.debug(f"Chunks: {chunks}")

    # then, we need to translate the chunks
    translated_chunks = []

    logger.info(f"Translating {len(chunks)} chunks")
    for chunk in tqdm(chunks):
        translated_chunk = translate_chunk(chunk, chain)
        logger.debug(f"Translated chunk: {translated_chunk}")

        # write the output to a file
        with open(output_path, "a") as f:
            f.write(translated_chunk)

        translated_chunks.append(translated_chunk)

    return translated_chunks


def translate(
    text: dict or str,
    api_key: str, # "not-needed" for local models
    model_name: str = "gpt-3.5-turbo",
    base_url: str | None = None,
    output_path: str = "translation.txt",
    temperature: int = 0,
    chunk_size: int = 2048,
    system_message: str = SYS_TRANSLATION_MESSAGE,
    human_message: str = HMN_TRANSLATION_MESSAGE,
    target_language: str="N/A",
    notes: str="N/A",
):
    """
    Translate text.

    Args:
        text (dict or str): Text to be translated. Can be a string or a dictionary
            containing a list of pages.
        api_key (str): API key for the LLM provider.
        model_name (str): Name of the model to be used for translation.
        temperature (int): Temperature to be used for translation.
        chunk_size (int): Size of the chunks to be used for translation.
        system_message (str): System message to be used for translation.
        human_message (str): Human message to be used for translation.
        more_info (dict): Dictionary containing additional information to be used
            for translation.

    Returns:
        str: String of the translated text.
    """

    logger.info("Translating text...")
    translation = None

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
    
    # create the more info dictionary
    more_info = {
        "target_language": target_language,
        "notes": notes,
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
            temperature=temperature,
            base_url=base_url
        )
    except Exception as e:
        logger.error(f"Error while creating LLM: {e}")

    # output parser
    output_parser = None
    try:
        output_parser = StrOutputParser()
    except Exception as e:
        logger.error(f"Error while creating output parser: {e}")

    # create the chain
    chain = None
    if prompt and llm and output_parser:
        try:
            chain = prompt | llm | output_parser
        except Exception as e:
            logger.error(f"Error while creating chain: {e}")
    else:
        logger.error("Could not create chain")

    # see if output_filepath exists, if not, create it
    if not os.path.exists(output_path):
        try:
            with open(output_path, "w") as f:
                f.write("")
        except Exception as e:
            logger.error(f"Error while creating output file: {e}")

    # if the file does exist, clear it
    else:
        try:
            with open(output_path, "w") as f:
                f.write("")
        except Exception as e:
            logger.error(f"Error while clearing output file: {e}")
    
    # get the tokenizer
    tokenizer = None
    try:
        tokenizer = get_tokenizer(model_name)
    except Exception as e:
        logger.error(f"Error while getting tokenizer: {e}")

    # translate the chunks
    translated_chunks = None
    if chain:
        try:
            translated_chunks = translate_chunks(text, chain, model_name, chunk_size, output_path, tokenizer)
        except Exception as e:
            logger.error(f"Error while translating chunks: {e}")

    # concatenate the chunks
    if translated_chunks:
        try:
            translation = " ".join(translated_chunks)
        except Exception as e:
            logger.error(f"Error while concatenating chunks: {e}")

    return translation
