import streamlit as st
from pdfreader import PDFReader
from scraper import WebReader
import os 
#import asyncio

#path to model directories
models_dir = 'models'
model_files = os.listdir(models_dir)
# Filter out any non-file entities, assuming you only want files
model_files = [file for file in model_files if os.path.isfile(os.path.join(models_dir, file))]

# Set the title for the Streamlit app
st.title("🧬 NASA OSDR Bot")

# Get model
mistral_model = st.sidebar.selectbox(
    "Select Model", options=model_files
)

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

# Add a text input box in the sidebar for website URL
study_id = st.sidebar.text_input("Enter OSDR ID:")


# Set assistant_type in session state
#if "mistral_model" not in st.session_state:
#    st.session_state["mistral_model"] = mistral_model
# Restart the assistant if assistant_type has changed
#elif st.session_state["mistral_model"] != mistral_model:
#    st.session_state["mistral_model"] = mistral_model
#    restart_assistant()

# Handle the uploaded file
if uploaded_file:
    pdfreader = PDFReader(uploaded_file, mistral_model)
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

# Sidebar for Initial Context
initial_context = st.sidebar.text_area("Initial Context", "Enter your text here...", height=300)
#st.sidebar.write("You've entered:", initial_context)
# Function to convert freeform text to JSON using LLM (mock function)
def text_to_json_with_llm(text):
    # This is where you'd send the text to the LLM with specific instructions.
    # For demonstration, we'll fabricate a response assuming the LLM parsed the text successfully.
    # Replace this with actual LLM integration code.
    response = {
        "title": "Example Study on AI",
        "authors": ["Jane Doe", "John Smith"],
        "abstract": "This study explores the impact of AI on global industries..."
    }
    return response

if st.sidebar.button('Convert to JSON'):
    # Convert the initial context to JSON using the LLM
    json_response = text_to_json_with_llm(initial_context)

    # Assuming you want to display the JSON in a readable format
    # json_formatted_str = json.dumps(json_response, indent=2)

    # Display the JSON response
    st.sidebar.text("JSON Output:")
    st.sidebar.json(json_response)
