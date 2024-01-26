# CORRECTION CONSTANTS
SYS_CORRECTION_MESSAGE = """You are a digitization specialist tasked with correcting OCR errors in digitized texts."""

HMN_CORRECTION_MESSAGE = """### Instructions:
The text below is a snippet from a digitized text. Your job is to carefully read the text and faithfully correct the OCR. This means keeping in mind the context of the text as you do your job. You have the following additional information about the source text, if available:
- Source filename: {filename}
- Source extension: {extension}
- Source filetype: {filetype}
- OCR software: {ocr_software}
- Image preprocessing software: {image_preprocessing_software}
- Publication date: {date}
- Language: {language}
- Comments: {comments}

If you would like to add comments, please put them under a heading like this:

### Comments:
This is a comment.



### Snippet:
{text}

### Response:
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

DEFAULT_CORRECTION_KWARGS = {
    "filename": "N/A",
    "extension": "PDF",
    "filetype": "N/A",
    "ocr_software": "pytesseract",
    "image_preprocessing_software": "opencv",
    "date": "N/A",
    "language": "N/A",
    "comments": "N/A",
}

# COLLATION CONSTANTS
SYS_COLLATION_MESSAGE = """
You are a detail-oriented content editor who is tasked with formatting OCR
text in a particular way for a client.
"""

HMN_COLLATION_MESSAGE = """INSTRUCTIONS FOR FORMATTING OCR TEXT
You will be given a snippet of text from a digitized text as well as the formatted text immaediately before it for context if available. Your job is to carefully read the texts and adjust the format of the target text according to the client's needs and what makes sense. The client has set the following criteria for formatting the text:
- Remove headers and footers: {remove_headers_and_footers}
- Remove page numbers: {remove_page_numbers}
- Remove excess space: {remove_excess_space}
- Remove empty lines: {remove_empty_lines}
- Remove line breaks: {remove_line_breaks}
- Remove word breaks: {remove_word_breaks}
- Section annotation: {add_section_tags}

DEFINITIONS AND EXAMPLES
- Headers and footers include things like source urls, abbreviated titles, etc.
- Excess space includes things like extra spaces between words, at the end of lines, between paragraphs, etc.
- Empty lines include lines that are completely empty, as well as lines that only contain whitespace.
- Line breaks are breaks caused by prior formatting and often break up sentences.
- When line breaks occur, sometimes words are broken up and hyphenated; these are word breaks.
- Section annotation involves adding tags to the text to indicate high level text layout.
- - For simplicity, annotation only involves tagging titles and named sections.
- - Annotation format is as follows: <HEADER>Section Name</HEADER>

VERY IMPORTANT NOTE
ONLY output the formatted version of the target text. Do NOT include anything else in your output; no commentary, no notes, nothing else.
If you absolutely have to add comments, please put them in double brackets like this: [[This is a comment.]].

ADDITIONAL CONTEXT
Here is the previously formatted text for additional context, if available:

{previous_text}

TEXT TO FORMAT

{text}
"""

DEFAULT_COLLATION_KWARGS = {
    "remove_headers_and_footers": True,
    "remove_page_numbers": True,
    "remove_excess_space": True,
    "remove_empty_lines": False,
    "remove_line_breaks": False,
    "remove_word_breaks": True,
    "add_section_tags": True,
    "keep_page_breaks": True,
}

# IMAGE PROCESSING CONSTANTS
DEFAULT_IMAGE_KWARGS = {
    "grayscale": True,
    "resize": False,
    "threshold": True,
    "deskew": False,
    "dilate_and_erode": False,
    "blur": False,
}

# TRANSLATION CONSTANTS
SYS_TRANSLATION_MESSAGE = """
You are a translator tasked with translating a digitized text.
"""

HMN_TRANSLATION_MESSAGE = """
The text below is a snippet from a digitized text. Your job is to carefully
read the text and faithfully translate it into the target language: {target_language}.
This means keeping in mind the context of the text as you do your job. You have the
following additional information about the source text, if available:
- Source title: {title}
- Source author: {author}
- Notes: {notes}

With this context in mind, please carefully read the following text, translate it,
and only return the translated text:

{text}
"""

DEFAULT_TRANSLATION_KWARGS = {
    "target_language": "N/A",
    "title": "N/A",
    "author": "N/A",
    "notes": "N/A",
}
