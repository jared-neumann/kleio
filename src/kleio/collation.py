# collating functions
# third party imports
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# internal imports
from kleio.general_utils import (
    setup_logger,
)

from kleio.llm_utils import (
    create_openai_prompt,
    create_openai_llm,
    parse_chunks,
)

from kleio.constants import (
    SYS_COLLATION_MESSAGE,
    HMN_COLLATION_MESSAGE
)

# set up logging
logger = setup_logger(__name__)


def collate_chunk(chunk: str, chain, previous_chunk: str):
    """
    Collate a chunk of text.
    """

    output = chain.invoke(
        {
            "previous_text": previous_chunk,
            "text": chunk,
        }
    )

    return output


def collate_chunks(chain, chunks: list):
    """
    Collate a list of chunks into a single text.
    """

    # we need to iterate through the chunks
    # and get the corrections
    # then concatenate the corrections
    # and return the text
    collated_text = ""

    previous_chunk = "N/A"
    for chunk in tqdm(chunks):
        output = collate_chunk(chunk, chain, previous_chunk)
        previous_chunk = output
        collated_text += output

    return collated_text


def collate(
    pages: list,
    api_key: str,
    model_name: str,
    llm_provider: str,
    base_url: str | None = None,
    temperature: int = 0,
    chunk_size: int = 1024,
    system_message: str = SYS_COLLATION_MESSAGE,
    human_message: str = HMN_COLLATION_MESSAGE,
    remove_headers_and_footers: bool = True,
    remove_page_numbers: bool = True,
    remove_excess_space: bool = True,
    remove_empty_lines: bool = False,
    remove_line_breaks: bool = False,
    remove_word_breaks: bool = True,
    add_section_tags: bool = True,
    keep_page_breaks: bool = True,
):
    """
    Collate a list of pages into a single text given a few formatting options.
    """

    # concatenate the pages into a single string
    # separate by <PAGEBREAK>
    text = "\n<PAGEBREAK>\n".join(pages)

    # convert collation args to more_info dict to put into prompt
    more_info = {
        "remove_headers_and_footers": remove_headers_and_footers,
        "remove_page_numbers": remove_page_numbers,
        "remove_excess_space": remove_excess_space,
        "remove_empty_lines": remove_empty_lines,
        "remove_line_breaks": remove_line_breaks,
        "remove_word_breaks": remove_word_breaks,
        "add_section_tags": add_section_tags,
        "keep_page_breaks": keep_page_breaks,
    }

    # we will need to chunk the text
    # then format each chunk
    # then concatenate the chunks
    # then return the text
    chunks = parse_chunks(
        text=text,
        chunk_size=chunk_size,
        model_name=model_name,
    )

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
            base_url=base_url,
            temperature=temperature,
        )
    else:
        logger.error("LLM provider not supported")
        return None

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    collated_text = collate_chunks(chain, chunks, model_name)

    return collated_text
