import os

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Qwen VLM Inference", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Qwen Visual Reasoning")
st.sidebar.header("Settings")
prompt_template = st.sidebar.text_area(
    "System Prompt",
    "Solve this task step by step. Wrap your reasoning in <think></think> and put the final answer in <answer></answer>.",
)

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
            uploaded_file.seek(0)
            try:
                response = requests.post(
                    f"{API_URL}/get_answer",
                    files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"prompt": f"{prompt_template}\n\n{user_input}"},
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not reach the inference API at {API_URL}: {exc}")
            else:
                if result.get("think"):
                    with st.expander("View Chain of Thought"):
                        st.write(result["think"])

                st.subheader("Final Answer")
                if result.get("answer"):
                    st.success(result["answer"])
                else:
                    st.write(result.get("raw", ""))  # Fallback if tags are missing
