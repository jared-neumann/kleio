# CORRECTION CONSTANTS
SYS_CORRECTION_MESSAGE = """You are a digitization specialist tasked with replacing poor OCR text with corrected text."""

HMN_CORRECTION_MESSAGE = """The text below is a snippet from a digitized text. Your job is to carefully read the text and faithfully correct the OCR. This means keeping in mind the context of the text as you do your job. You have the following additional information about the source text, if available:
- Source filename: {filename}
- Source filetype: {filetype}
- OCR software: {ocr_software}
- Image preprocessing software: {image_preprocessing_software}
- Publication date: {date}
- Language: {language}
- Comments: {comments}

Please correct the following text:
{text}

Please return only your corrected version that snippet of text as if it had to seemlessly replace the original OCR text:
"""

EXAMPLE_CORRECTION_TEXT = """
PREFACE.

Havine concluded our Firft Volume, we
would be deficient in gratitude did we not return
thinks to the Public, in general, for the favourable
reception our labours have experienced; and to
thofe Scientific Gentlemen, in particular, who have
aflifted us with Communications, as well as Hints
refpecting the future condudting of the Work,

As the grand Object of it is to diffufe Philofo-
phical Knowledge among every Clafs of Society,
and to give the Public as early an Account as pof-
fible of every thing new or curious in the fcientific
World, both at Home and on the Continent, we
flatter ourfelves with the hope that the fame liberal
Patronage we have hitherto experienced will be
continued; and that Scientific Men will afford us
that Support and Affiftance which they may think
our Attempt entitled to. Whatever may be our
future Succefs, no Exertions fhall be wanting on our
part to render the Work ufeful to Society, and efpe-
cially to the Arts and Manufactures of Great Britain
which, as is well known, have been much improved
by the great Progrefs that has lately been made in
various Branches of the Philofophical Sciences. ~~

A2
"""

# COLLATION CONSTANTS
SYS_COLLATION_MESSAGE = """You are a detail-oriented content editor who is tasked with formatting OCR text in a particular way for a client."""

HMN_COLLATION_MESSAGE = """INSTRUCTIONS FOR FORMATTING OCR TEXT
You will be given a snippet of text from a digitized text as well as the formatted text immaediately before it for context if available. Your job is to carefully read the texts and adjust the format of the target text according to the client's needs and what makes sense. The client has set the following criteria for formatting the text:
- Remove headers and footers: {remove_headers_and_footers}
- Remove page numbers: {remove_page_numbers}
- Remove excess space: {remove_excess_space}
- Remove empty lines: {remove_empty_lines}
- Remove line breaks: {remove_line_breaks}
- Remove word breaks: {remove_word_breaks}
- Add section annotations: {add_section_tags}

DEFINITIONS AND EXAMPLES
- Headers and footers include things like source urls, abbreviated titles, etc.
- Excess space includes things like extra spaces between words, at the end of lines, between paragraphs, etc.
- Empty lines include lines that are completely empty, as well as lines that only contain whitespace.
- Line breaks are breaks caused by prior formatting and often break up sentences.
- When line breaks occur, sometimes words are broken up and hyphenated; these are word breaks.
- Section annotation involves adding a basic tag to identifiable section headings.
- - ONLY DO THIS IF IT IS SET TO TRUE.
- - Annotation ONLY involves tagging titles and named sections.
- - Annotation format is as follows: <SECTION>Example</SECTION>

ADDITIONAL NOTES
- ONLY output the formatted version of the target text. Do NOT include anything else in your output; no commentary, no notes, nothing else.
- If you absolutely have to add comments, please put them in double brackets like this: [[This is a comment.]].
- Go ahead and correct any remaining OCR errors, such as removing junk strings, removing other people's commentary on the text, etc.

ADDITIONAL CONTEXT
Here is the previously formatted text for additional context, if this is not the first or only chunk of the document:

{previous_text}

TEXT TO FORMAT
{text}
"""

# TRANSLATION CONSTANTS
SYS_TRANSLATION_MESSAGE = """You are a translator tasked with translating a digitized text."""

HMN_TRANSLATION_MESSAGE = """INSTRUCTIONS
The text below is a snippet from a digitized text. Your job is to carefully
read the text and faithfully translate it into the target language: {target_language}.
This means keeping in mind the context of the text as you do your job.

NOTES (if available):
{notes}

With this context in mind, please carefully read the following text, translate it,
and only return the translated text:

{text}
"""
