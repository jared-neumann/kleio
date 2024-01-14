import streamlit as st

st.session_state['uploaded_file'] = None
st.session_state['ocr_results'] = None
st.session_state['correction_results'] = None
st.session_state['collation_results'] = None

# create a title
st.title("Kleio")

# create a subtitle
st.markdown("A tool for extracting text, correcting errors, collating pages, and more.")

# put the about section in the sidebar
st.sidebar.title("About")
st.sidebar.info(
    """
    This is a user interface for the Kleio library.
    Please upload a file to get started.
    Then, use each form to choose the action you want to perform:
    - Extract text
    - Correct errors
    - Collate pages
    """
)

# separate forrms in the main section
st.subheader("UPLOAD FILE")


st.subheader("EXTRACT TEXT")


st.subheader("CORRECT ERRORS")


st.subheader("COLLATE PAGES")
