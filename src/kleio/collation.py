# collating functions
import os

# third party imports
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# internal imports
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
    SYS_COLLATION_MESSAGE,
    HMN_COLLATION_MESSAGE
)

# set up logging
logger = setup_logger(__name__)


def collate_chunk(chunk: str, chain, previous_chunk: str, output_path: str) -> str:
    """
    Collate a chunk of text.
    """

    if previous_chunk != "N/A":
        # get the last 50 characters of the previous chunk as a lead-in
        lead_in = previous_chunk[-50:]
        # format the lead-in so it make sense
        lead_in = "..." + lead_in + "..."

    else:
        lead_in = "This is the first or only chunk."

    # get the corrections
    logger.debug(f"Collating chunk: {chunk}\n\nLead-in: {lead_in}")
    output = chain.invoke({"previous_text": lead_in, "text": chunk})

    # write the output to a file
    with open(output_path, "a") as f:
        f.write(output)

    return output


def collate_chunks(chain, chunks: list, output_path: str):
    """
    Collate a list of chunks into a single text.
    """

    logger.info(f"Collating {len(chunks)} chunks")

    # we need to iterate through the chunks
    # and get the corrections
    # then concatenate the corrections
    # and return the text
    collated_text = []
    previous_chunk = "N/A"
    output = None
    for chunk in tqdm(chunks):
        # get the corrections
        output = collate_chunk(chunk, chain, previous_chunk, output_path)

        # set the previous chunk
        previous_chunk = output

        # append the output to the collated text
        collated_text.append(output)
    
    # concatenate the collated text
    collated_text = "".join(collated_text)
    logger.debug(f"Collated text: {collated_text}")

    return collated_text


def collate(
    pages: list,
    api_key: str, # "not-needed" for local models
    model_name: str = "gpt-3.5-turbo-16k",
    base_url: str | None = None,
    output_path: str = "collation.txt",
    temperature: int = 0,
    chunk_size: int = 2048,
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

    # get tokenizer
    tokenizer = None
    try:
        tokenizer = get_tokenizer(model_name)
    except Exception as e:
        logger.error(f"Error while getting tokenizer: {e}")
    
    chunks = None
    try:
        chunks = parse_chunks(
            text=text,
            chunk_size=chunk_size,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        logger.debug(f"Chunks: {chunks}")
    except Exception as e:
        logger.error(f"Error while parsing chunks: {e}")

    # create the prompt
    prompt = None
    try:
        prompt = create_prompt(
            system_message=system_message,
            human_message=human_message,
            more_info=more_info,
        )
        logger.debug(f"Prompt: {prompt}")
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

    # create an output file if it doesn't exist
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            f.write("")
    else: # erase the contents of the file if it exists
        with open(output_path, "r+") as f:
            f.truncate(0)

    # correct the text
    collated_text = None
    if chain:
        try:
            collated_text = collate_chunks(chain, chunks, output_path=output_path)
        except Exception as e:
            logger.error(f"Error while collating text: {e}")

    return collated_text
