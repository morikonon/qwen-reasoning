import streamlit as st
from PIL import Image
import re
from model import load_model_and_processor, run_inference

st.set_page_config(page_title="Qwen VLM Inference", layout="wide")

# Constants
BASE_MODEL = "Qwen/Qwen3.5-0.8b" #
ADAPTER_DIR = "./weights"

@st.cache_resource
def get_model():
    return load_model_and_processor(BASE_MODEL, ADAPTER_DIR)

model, processor = get_model()

st.title("Qwen Visual Reasoning")
st.sidebar.header("Settings")
prompt_template = st.sidebar.text_area("System Prompt", 
    "Solve this task using <think> ... </think> and put the answer into <answer> </answer>")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    user_input = st.text_input("What would you like to know about this image?")
    
    if st.button("Generate") and uploaded_file and user_input:
        with st.spinner("Thinking..."):
            full_response = run_inference(model, processor, image, f"{prompt_template}\n\n{user_input}")
            
            # Parsing the Thinking process vs Answer
            think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
            
            if think_match:
                with st.expander("View Chain of Thought"):
                    st.write(think_match.group(1).strip())
            
            st.subheader("Final Answer")
            if answer_match:
                st.success(answer_match.group(1).strip())
            else:
                st.write(full_response) # Fallback if tags are missing