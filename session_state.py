import os
from utils.prepare_vectordb import get_vectorstore
from datetime import datetime

def initialize_session_state_variables(st):
    """
    Initialize session state variables for the Streamlit application

    Parameters:
    - st (streamlit.delta_generator.DeltaGenerator): Streamlit's DeltaGenerator object used for rendering elements
    """
    # Get the list of uploaded documents
    upload_docs = os.listdir("docs")
    
    # List of session state variables to initialize
    variables_to_initialize = ["chat_history", "uploaded_pdfs", "processed_documents", "vectordb", "previous_upload_docs_length", "session_id"]
    
    # Iterate over the variables and initialize them if not present in the session state 
    for variable in variables_to_initialize:
        if variable not in st.session_state:
            if variable == "processed_documents":
                # Set to the name of the files present in the docs folder
                st.session_state.processed_documents = upload_docs
            elif variable == "vectordb":
                # Set to None initially, then use the existing vector database if present
                st.session_state.vectordb = get_vectorstore(upload_docs, from_session_state=True)
            elif variable == "previous_upload_docs_length":
                # Set to the quantity of documents in the docs folder during app startup
                st.session_state.previous_upload_docs_length = len(upload_docs)
            elif variable == "session_id":
                # Initialize session_id if not already present
                st.session_state.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            else:
                # Initialize other session variables as empty lists
                st.session_state[variable] = []
