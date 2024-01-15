import os
import pytest

import kleio.llm_utils as llm_utils


def test_create_openai_prompt():
    """
    Test the create_openai_prompt function in llm_utils.py.

    Checks that the function returns a prompt.
    """

    prompt = llm_utils.create_openai_prompt(
        "System message test", "Human message {test}", {"test": "test"}
    )

    assert prompt is not None

pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="No API key")
def test_create_openai_llm():
    """
    Test the create_openai_llm function in llm_utils.py.

    Checks that the function returns an LLM.
    """

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    llm = llm_utils.create_openai_llm(
        OPENAI_API_KEY, "gpt-3.5-turbo", temperature=0
    )

    assert llm is not None


def test_parse_chunks():
    """
    Test the parse_chunks function in llm_utils.py.

    Checks that the function returns a list of chunks.
    """

    chunks = llm_utils.parse_chunks(
        "This is a test sentence.", chunk_size=1024, model_name="gpt-3.5-turbo"
    )

    assert chunks is not None


def test_sentencize():
    """
    Test the sentencize function in llm_utils.py.

    Checks that the function returns a list of sentences.
    """

    sentences = llm_utils.sentencize("This is a test sentence.")

    assert sentences is not None


def test_parse_sentence_chunks():
    """
    Test the parse_sentence_chunks function in llm_utils.py.

    Checks that the function returns a list of chunks.
    """

    chunks = llm_utils.parse_sentence_chunks(
        "This is a test sentence.", model_name="gpt-3.5-turbo", chunk_size=1024
    )

    assert chunks is not None
