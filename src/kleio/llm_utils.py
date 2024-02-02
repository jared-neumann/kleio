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
from transformers import AutoTokenizer, GPT2TokenizerFast
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

# internal modules
from kleio.general_utils import (
    setup_logger,
)

# set up logging
logger = setup_logger(__name__)

def get_tokenizer(model_name: str):
    """
    Get a tokenizer for a given model.

    Args:
        model_name (str): Name of the model to be used.

    Returns:
        AutoTokenizer: Tokenizer for the model.
    """
    tokenizer = None
    if "gpt" in model_name:
        model_name = "Xenova/gpt-3.5-turbo"
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error while loading tokenizer: {e}")
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.error(f"Error while loading tokenizer: {e}")

    return tokenizer

def create_prompt(system_message: str, human_message: str, more_info: dict):
    """
    Create a ChatOpenAI prompt.
    
    Args:
        system_message (str): establishes the role of the llm
        human_message (str): provides context for the task and the text to
            be corrected
        more_info (dict): dictionary of additional information about the text

    Returns:
        ChatPromptTemplate: ChatOpenAI prompt
    """

    logger.info("Creating ChatOpenAI prompt...")
    cht_prompt = None

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

def create_llm(
        api_key: str="not-needed",
        model_name: str="not-used",
        base_url: str|None=None,
        temperature: int=0
) -> ChatOpenAI:
    """
    Create an LLM from ChatOpenAI.

    Args:
        api_key (str): API key for the LLM provider.
        model_name (str): Name of the model to be used.
        base_url (str): Base URL for the LLM provider.
        temperature (int): Temperature to be used.

    Returns:
        ChatOpenAI: LLM instance.
    """

    logger.info("Creating OpenAI LLM...")
    llm = None

    if base_url:
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Error while creating Local LLM: {e}")
            logger.info("Make sure you have the local server running and the correct variables, or use another LLM provider/api key")
    else:
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
            )
        except Exception as e:
            logger.error(f"Error while creating OpenAI LLM: {e}")

    return llm


def parse_chunks(
    text: str,
    chunk_size: int = 2048,
    model_name: str = "Xenova/gpt-3.5-turbo",
    tokenizer: AutoTokenizer = None,
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

    chunks = None

    if tokenizer:
        # tokenize the text
        try:
            text_tokenized = tokenizer.tokenize(text)
        except Exception as e:
            logger.error(f"Error while tokenizing text: {e}")

        # chunk the text
        try:
            chunks = [
                tokenizer.convert_tokens_to_string(text_tokenized[i : i + chunk_size])
                for i in range(0, len(text_tokenized), chunk_size)
            ]
        except Exception as e:
            logger.error(f"Error while chunking text: {e}")
        
    else:
        # nltk tokenization
        logger.info("No tokenizer loaded. Proceeding with NLTK tokenization...")
        try:
            words = word_tokenize(text)
        except Exception as e:
            logger.error(f"Error while tokenizing text: {e}")
        
        try:
            chunks = [
                " ".join(words[i : i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]
        except Exception as e:
            logger.error(f"Error while chunking text: {e}")

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
    text: str, chunk_size: int = 2048, model_name: str = "Xenova/gpt-3.5-turbo", tokenizer = None
):
    """
    Parse a text into chunks of sentences.

    Args:
        text (str): Text to be parsed into chunks.

    Returns:
        list: List of chunks.
    """

    logger.info("Parsing text into sentence chunks...")
    chunks = []

    if text:
        # split the text into sentences
        sentences = sentencize(text)

    if tokenizer:
        try:
            # tokenize the sentences
            tokenized_sentences = [tokenizer.tokenize(sent) for sent in sentences]
        except Exception as e:
            logger.error(f"Error while tokenizing sentences: {e}")
            return None
    else:
        logger.info("No tokenizer loaded. Proceeding with NLTK tokenization...")
        try:
            # nltk tokenization
            tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        except Exception as e:
            logger.error(f"Error while tokenizing sentences: {e}")
            return None

    # chunk the sentences
    chunk = []
    for sent in tokenized_sentences:
        if len(" ".join(chunk)) + len(" ".join(sent)) > int(chunk_size):
            chunks.append(" ".join(chunk))
            chunk = []
        chunk.append(sent)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks if len(chunks) > 0 else None
