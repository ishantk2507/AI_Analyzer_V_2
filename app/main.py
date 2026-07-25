"""
Streamlit UI for the local agentic data analysis system.

Single-page interface for dataset upload, chat-based querying,
and result/chart display. Runs on localhost only.
"""

import streamlit as st
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import base64

from app.config import MODEL_PATH, DATA_MOUNT_DEFAULT, SCRATCH_MOUNT_DEFAULT
from app.agent.graph import AgentGraph


# Page configuration
st.set_page_config(
    page_title="Local Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit branding and footer via CSS
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model_path() -> str:
    """Return the configured model path (cached)."""
    return MODEL_PATH


def init_session_state():
    """Initialize session state variables."""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'data_dir' not in st.session_state:
        st.session_state.data_dir = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None


def cleanup_agent():
    """Clean up the current agent session."""
    if st.session_state.agent:
        try:
            st.session_state.agent.cleanup()
        except Exception:
            pass
        st.session_state.agent = None
    
    if st.session_state.data_dir:
        try:
            shutil.rmtree(st.session_state.data_dir, ignore_errors=True)
        except Exception:
            pass
        st.session_state.data_dir = None
    
    st.session_state.chat_history = []
    st.session_state.dataset_name = None


def create_data_directory() -> Path:
    """Create a temporary directory for uploaded data."""
    temp_dir = Path(tempfile.mkdtemp(prefix='data_analyst_'))
    return temp_dir


def handle_file_upload(uploaded_files) -> bool:
    """
    Handle file upload and initialize agent.
    
    Returns True if successful.
    """
    if not uploaded_files:
        return False
    
    data_dir = create_data_directory()
    
    for uploaded_file in uploaded_files:
        # Save uploaded file to data directory
        file_path = data_dir / uploaded_file.name
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
    
    # Initialize agent
    try:
        model_path = get_model_path()
        st.session_state.agent = AgentGraph(
            data_dir=str(data_dir),
            model_path=model_path if Path(model_path).exists() else None,
        )
        st.session_state.data_dir = data_dir
        st.session_state.dataset_name = ', '.join(f.name for f in uploaded_files)
        return True
    except Exception as e:
        cleanup_agent()
        st.error(f"Failed to initialize agent: {e}")
        return False


def render_chat_message(role: str, content: str, artifacts: list = None):
    """Render a chat message with optional artifacts."""
    with st.chat_message(role):
        st.write(content)
        
        if artifacts:
            for artifact_path in artifacts:
                if artifact_path.endswith('.png'):
                    # Try to load and display the chart
                    if st.session_state.agent:
                        try:
                            img_data = st.session_state.agent.get_artifact(artifact_path)
                            if img_data:
                                st.image(img_data, use_container_width=True)
                        except Exception:
                            pass


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Sidebar for dataset management
    with st.sidebar:
        st.header("📁 Dataset")
        
        if st.session_state.dataset_name:
            st.success(f"Connected: {st.session_state.dataset_name}")
            if st.button("Disconnect & New Dataset", use_container_width=True):
                cleanup_agent()
                st.rerun()
        else:
            st.info("Upload a dataset to begin")
            
            uploaded_files = st.file_uploader(
                "Upload CSV, Parquet, or XLSX files",
                type=['csv', 'parquet', 'xlsx'],
                accept_multiple_files=True,
            )
            
            if uploaded_files:
                if st.button("Connect Dataset", use_container_width=True, type='primary'):
                    with st.spinner("Loading dataset..."):
                        if handle_file_upload(uploaded_files):
                            st.success("Dataset connected!")
                            st.rerun()
        
        st.divider()
        
        # Configuration info
        st.subheader("⚙️ Configuration")
        st.caption(f"Model: {MODEL_PATH.split('/')[-1] if MODEL_PATH else 'Not set'}")
        st.caption("Mode: Fully offline, local execution")
    
    # Main chat area
    st.title("🤖 Local Data Analyst")
    
    if not st.session_state.agent:
        st.markdown("""
        ### Welcome!
        
        Upload a dataset (CSV, Parquet, or XLSX) using the sidebar to get started.
        
        Once connected, you can ask questions in plain language like:
        - "What are the top 5 categories by sales?"
        - "Show me the distribution of customer ages"
        - "Is there a correlation between price and quantity sold?"
        
        The agent will explore your data, run analysis in an isolated sandbox,
        and provide findings with business-relevant interpretations.
        """)
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        render_chat_message(
            message['role'],
            message['content'],
            message.get('artifacts', [])
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt,
            'artifacts': []
        })
        
        with st.chat_message('user'):
            st.write(prompt)
        
        # Run agent
        with st.chat_message('assistant'):
            with st.spinner("Analyzing..."):
                try:
                    result = st.session_state.agent.run(prompt)
                    
                    response = result.get('final_response', 'Analysis completed.')
                    artifacts = result.get('artifacts', [])
                    errors = result.get('errors', [])
                    
                    # Display response
                    st.write(response)
                    
                    # Display any errors encountered
                    if errors:
                        with st.expander("⚠️ Notes from analysis"):
                            for error in errors:
                                st.caption(f"- {error}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response,
                        'artifacts': artifacts
                    })
                    
                except Exception as e:
                    error_msg = f"Error during analysis: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': error_msg,
                        'artifacts': []
                    })
        
        st.rerun()


if __name__ == '__main__':
    main()
