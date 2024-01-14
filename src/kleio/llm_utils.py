# llm utility functions
# third-party modules
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import tiktoken

# internal modules
from kleio.general_utils import (
    setup_logger,
)

# set up logging
logger = setup_logger(__name__)


def create_openai_prompt(system_message: str, human_message: str, more_info: dict):
    """
    Create a ChatOpenAI prompt.

    Args:
        system_message (str): establishes the role of the llm
        human_message (str): provides context for the task and the text to
            be corrected
        more_info (dict): dictionary of additional information about the text
    """
    logger.info("Creating OpenAI prompt for OCR correction task")

    # set up the prompt
    # first, the system message
    try:
        sys_prompt = SystemMessagePromptTemplate.from_template((system_message))
    except Exception as e:
        logger.error(f"Error while creating system message prompt: {e}")

    # then, the human message
    try:
        hmn_prompt = HumanMessagePromptTemplate.from_template(
            (human_message), partial_variables=more_info
        )
    except Exception as e:
        logger.error(f"Error while creating human message prompt: {e}")

    # put it together into a chat prompt
    try:
        cht_prompt = ChatPromptTemplate.from_messages([sys_prompt, hmn_prompt])
    except Exception as e:
        logger.error(f"Error while creating chat prompt: {e}")
        return None

    return cht_prompt


def create_openai_llm(api_key: str, model_name: str, temperature: int):
    """
    Create an LLM from OpenAI.
    """
    logger.info("Creating OpenAI LLM")

    # create the llm
    try:
        llm = ChatOpenAI(
            openai_api_key=api_key, model_name=model_name, temperature=temperature
        )
    except Exception as e:
        logger.error(f"Error while creating OpenAI LLM: {e}")
        return None

    return llm


def parse_chunks(
    text: str,
    chunk_size: int = 1024,
    model_name: str = "gpt-3.5-turbo",
):
    """
    Parse a text into chunks of a given size according to
    a given encoding.

    Args:
        text (str): Text to be parsed into chunks.
        chunk_size (int): Size of the chunks.
        model_name (str): Name of the model to be used for encoding.

    Returns:
        list: List of chunks.
    """
    enc = None
    if "gpt" in model_name:
        enc = "r50k_base"
    
    if enc:
        encoding = tiktoken.get_encoding(enc)

    # encode the text
    try:
        text_encoding = encoding.encode(text)
    except Exception as e:
        logger.error(f"Error while encoding text: {e}")
        return None

    # chunk the text
    try:
        chunks = [
            encoding.decode(text_encoding[i : i + chunk_size])
            for i in range(0, len(text_encoding), chunk_size)
        ]
    except Exception as e:
        logger.error(f"Error while chunking text: {e}")
        return None

    return chunks
