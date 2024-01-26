# llm utility functions

# built-in modules
import subprocess

# third-party modules
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer
import tiktoken
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

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


def create_openai_llm(api_key: str, model_name: str, base_url: str|None=None, temperature: int=0):
    """
    Create an LLM from OpenAI.
    """
    logger.info("Creating OpenAI LLM")

    if base_url:
        try:
            llm = ChatOpenAI(
                api_key="not-needed",
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Error while creating Local LLM: {e}")
            return None
    else:
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
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
    tokenizer = None
    if "gpt" in model_name:
        enc = "r50k_base"
    elif model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    
    elif tokenizer:
        # tokenize the text
        try:
            text_tokenized = tokenizer.tokenize(text)
        except Exception as e:
            logger.error(f"Error while tokenizing text: {e}")
            return None

        # chunk the text
        try:
            chunks = [
                tokenizer.convert_tokens_to_string(text_tokenized[i : i + chunk_size])
                for i in range(0, len(text_tokenized), chunk_size)
            ]
        except Exception as e:
            logger.error(f"Error while chunking text: {e}")
            return None
        
    else:
        logger.error(f"Error while chunking text: no tokenizer or encoding found")
        return None

    return chunks


def sentencize(text: str):
    """
    Split a text into sentences.

    Args:
        text (str): Text to be split into sentences.

    Returns:
        list: List of sentences.
    """
    sentences = None
    logger.info("Splitting text into sentences")
    try:
        # spacy
        nlp = spacy.load("en_core_web_sm")
        sentences = [sent.text for sent in nlp(text).sents]
    except Exception as e:
        logger.warn(f"Error while loading spacy model: {e}")
        logger.info("Proceeding with NLTK sentence tokenizer")

        try:
            # nltk
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.error(f"Error while sentence tokenizing text: {e}")
            return None

    return sentences


def parse_sentence_chunks(
    text: str, model_name="gpt-3.5-turbo", chunk_size: int = 1024
):
    """
    Parse a text into chunks of sentences.

    Args:
        text (str): Text to be parsed into chunks.

    Returns:
        list: List of chunks.
    """
    if text:
        # split the text into sentences
        sentences = sentencize(text)

        # encode the sentences
        enc = None
        if "gpt" in model_name:
            enc = "r50k_base"

        if enc:
            encoding = tiktoken.get_encoding(enc)

        try:
            encoded_sentences = [encoding.encode(sent) for sent in sentences]
        except Exception as e:
            logger.error(f"Error while encoding sentences: {e}")
            return None

        # chunk the sentences
        chunks = []
        chunk = []

        for encoded_sentence in encoded_sentences:
            if len(chunk) + len(encoded_sentence) > chunk_size:
                chunks.append(chunk)
                chunk = []
            chunk += encoded_sentence
        chunks.append(chunk)

        # decode the chunks
        try:
            chunks = [encoding.decode(chunk) for chunk in chunks]
        except Exception as e:
            logger.error(f"Error while decoding chunks: {e}")
            return None

        return chunks
