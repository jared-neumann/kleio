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
    HMN_COLLATION_MESSAGE,
    DEFAULT_COLLATION_KWARGS,
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


def collate_chunks(chain, chunks: list, model_name: str):
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
    more_info: dict = DEFAULT_COLLATION_KWARGS,
):
    """
    Collate a list of pages into a single text given a few formatting options.
    """

    # concatenate the pages into a single string
    # separate by <PAGEBREAK>
    text = "\n<PAGEBREAK>\n".join(pages)

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
