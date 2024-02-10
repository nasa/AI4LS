import streamlit as st
from pdfreader import PDFReader
from scraper import WebReader
#import asyncio


# Set the title for the Streamlit app
st.title("🧬 NASA OSDR Bot")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

# Add a text input box in the sidebar for website URL
study_id = st.sidebar.text_input("Enter OSDR ID:")


# Handle the uploaded file
if uploaded_file:
    pdfreader = PDFReader(uploaded_file)
    pdfreader.handlefileandingest()

# You can add additional logic here to handle the website URL
# For example:
if study_id:
    # Process the website URL as needed
    def construct_url(osdr_id):
        base_url = "https://osdr.nasa.gov/bio/repo/data/studies/"
        return base_url + study_id

    # Example usage
    #input_id = "OSD-609"
    website_url = construct_url(study_id)
    st.write("For more info: ", website_url)
    webreader = WebReader(website_url)
    webreader.loadweb()
