SYS_CORRECTION_MESSAGE = """
You are a helpful digitization specialist tasked with correcting OCR
errors in digitized texts.
"""

HMN_CORRECTION_MESSAGE = """
The text below is a snippet from a digitized text. Your job is to carefully
read the text and faithfullycorrect the OCR. This means keeping in mind the
context of the text as you do your job. You have the following additional
information about the source text, if available:
- Source filename: {filename}
- Source extension: {extension}
- Source filetype: {filetype}
- OCR software: {ocr_software}
- Image preprocessing software: {image_preprocessing_software}
- Publication date: {date}
- Language: {language}
- Comments: {comments}

With this context in mind, please carefully read the following text and
faithfully correct the OCR and only return the corrected text:

{text}
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

SYS_COLLATION_MESSAGE = """
You are a detail-oriented content editor who is tasked with formatting OCR
text in a particular way for a client.
"""

HMN_COLLATION_MESSAGE = """
You will be given a snippet of text from a digitized text as well as the formatted text
immaediately before it for context if available. Your job is to carefully read the texts
and adjust the format of the target text according to the client's needs and what makes
sense. The client has provided you with the following answers to a questionaire
for the end result:
- Would you like to remove headers and footers? {remove_headers_and_footers}
- What about page numbers, that sort of thing? {remove_page_numbers}
- And extra spaces? reduce? {remove_excess_space}
- Remove empty lines? {remove_empty_lines}
- Should we concatenate lines? {remove_line_breaks}
- Put words back together that were hyphenated across line-breaks? {remove_word_breaks}
- Add section tags? {add_section_tags}
- Keep page breaks? {keep_page_breaks}

If required by the client, tags should be added in the following format:
[SECTION_HEADER]Section header[/SECTION_HEADER]

Sections include things like titles, chapter headings, named subsections, etc.
But, you should ONLY use the tag: SECTION_HEADER. More specific tags are not
necessary.

Headers and footers include things like page numbers, page titles, other
random or unrelated or abbreviated text that appears at the top or bottom of a
page, etc.

Just output the formatted version of the target text. Here is the previously
formatted text for additional context, if available:

{previous_text}

And here is the target text to format:

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
