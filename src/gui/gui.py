import streamlit as st
import requests
from io import BytesIO
import docx2txt
import fitz  # PyMuPDF
import os
import time
import streamlit.components.v1 as components

response_code = 400
processing_time = 0

# Set the app to wide mode
st.set_page_config(layout="wide")

st.title("Công cụ tóm tắt văn bản")

compression_list = [f"{x * 10}% của văn bản gốc" for x in range(1, 10)]

model_dicts = {
    "Custom GPT 1 - 2 Billion": "2b",
    "Custom GPT 2 - 7 Billion": "7b",
}

compression_dicts = {
    "10% của văn bản gốc": 10,
    "20% của văn bản gốc": 20,
    "30% của văn bản gốc": 30,
    "40% của văn bản gốc": 40,
    "50% của văn bản gốc": 50,
    "60% của văn bản gốc": 60,
    "70% của văn bản gốc": 70,
    "80% của văn bản gốc": 80,
    "90% của văn bản gốc": 90,
}

# Initialize session state for storing assistant messages and result text
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "result_text" not in st.session_state:
    st.session_state.result_text = ""
if "compression_ratio" not in st.session_state:
    st.session_state.compression_ratio = 50  # Default value

# Split the page into two columns
col1, col2 = st.columns(2)

# Left column: User input and summary type selection
with col1:
    # Initialize prompt variable
    prompt = ""

    # Add file uploader for text files
    uploaded_file = st.file_uploader(
        "Tải lên một tệp văn bản (.txt, .docx, .dotx, .pdf)",
        type=["txt", "docx", "dotx", "pdf"],
    )

    # If a file is uploaded, process it based on its type
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        temp_filename = f"temp_file.{file_extension}"

        # Save uploaded file to a temporary location
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.read())

        # Process the file based on its extension
        if file_extension == "txt":
            prompt = open(temp_filename, "r", encoding="utf-8").read()
        elif file_extension in ["docx", "dotx"]:
            prompt = docx2txt.process(temp_filename)
        elif file_extension == "pdf":
            doc = fitz.open(temp_filename)
            prompt = ""
            for page in doc:
                prompt += page.get_text("text").replace("\n", "")
            doc.close()

        # Clean up temporary file
        os.remove(temp_filename)

    # Input text area
    prompt = st.text_area(
        "Nhập văn bản cần tóm tắt, tối đa 5000 ký tự",
        value=prompt,
        height=300,
        key="input_text_area",
        max_chars=5000,
    )

    # Add input for summary type selection below the text area
    summary_type = st.selectbox(
        "Chọn loại tóm tắt:", ("Tóm tắt ngắn gọn", "Tóm tắt chi tiết")
    )

    # Add input for compression ratio
    compression_ratio = st.selectbox(
        "Chọn tỷ lệ nén:",
        (
            compression_list[:6]
            if summary_type == "Tóm tắt ngắn gọn"
            else compression_list[3:]
        ),
        index=4 if summary_type == "Tóm tắt ngắn gọn" else 0,
    )

    st.session_state.compression_ratio = compression_ratio

    # Add model selection
    model_type = st.selectbox(
        "Chọn mô hình:", ("Custom GPT 1 - 2 Billion", "Custom GPT 2 - 7 Billion")
    )

    # Button to submit the text
    submit = st.button("Tóm tắt", type="primary")

# Right column: Display instructions and results
with col2:
    # Always show instructions
    st.markdown(
        """
        <div style="text-align:left; margin-top:20px;">
            <h3>Hướng dẫn sử dụng:</h3>
            <p>1. Nhập dữ liệu hoặc dán dữ liệu vào ô bên trái.</p>
            <p>2. Chọn tùy chọn: tóm tắt ngắn gọn hay tóm tắt chi tiết.</p>
            <p>3. Chọn tỷ lệ nén phù hợp.</p>
            <p>4. Chọn mô hình: 2b hoặc 7b.</p>
            <p>5. Bấm nút Tóm Tắt và nhận kết quả.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <hr style="border: 1px solid #ddd; margin-top: 10px; margin-bottom: 10px;">
        """,
        unsafe_allow_html=True,
    )

    # Display loading state and result
    result = ""
    if submit and prompt:
        model_use = model_dicts[model_type]
        start_time = time.time()
        with st.spinner("Đang xử lý..."):
            try:
                # Send request to API
                api_url = ("http://localhost:8080/backend/predict")
                # Replace with actual API URL

                headers = {
                    "Access-Control-Allow-Origin": "*",
                    "X-Requested-With": "XMLHttpRequest",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }

                data = {
                    "text": prompt[:5000],
                    "type": summary_type,
                    "model": model_use,
                    "compression": compression_dicts[compression_ratio],
                }

                response = requests.post(api_url, json=data, headers=headers)

                # Calculate processing time
                end_time = time.time()
                processing_time = end_time - start_time

                # Handle API response
                if response.status_code == 200:
                    response_code = 200
                    response_data = response.json()
                    result = response_data["data"]
                    st.toast(
                        "Hoàn thành", icon="✅"
                    )  # Using st.success for success message
                # else:
                #     result = "Error: Unable to summarize the text."
            except Exception as e:
                end_time = time.time()
                response_code = 400
                result = f"Error: {str(e)}"
                processing_time = end_time - start_time

            # Store the result in session state
            st.session_state.result_text = result

    # Display result if available with max height of 400px and scroll enabled
    if st.session_state.result_text:
        result_text = st.session_state.result_text.split("## Tóm tắt:")
        text_output = result_text[1].replace("\n", "")
        text_output = " ".join(text_output.split())
        st.markdown("### Tóm tắt:")
        st.markdown(
            f"""
                <div style="max-height: 400px; overflow-y: auto; padding: 10px;">
                    {text_output}</div>
            """,
            unsafe_allow_html=True,
        )

        # Copy to clipboard button with icon
        if response_code == 200:
            # Button with custom style and copy to clipboard functionality
            components.html(
                f"""
                <button id="copy-button" style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                ">
                    <i class="fas fa-copy"></i> Sao chép
                </button>
                <script src="https://unpkg.com/sweetalert/dist/sweetalert.min.js"></script>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/js/all.min.js"></script>
                <script>
                    document.getElementById('copy-button').addEventListener('click', () => {{
                        navigator.clipboard.writeText(`{text_output}`).then(() => {{
                            console.log('`{text_output}`')
                        }}, err => {{
                            console.log('`{text_output}` copy error: ' + err)
                        }});
                    }});
                </script>
                """,
                height=80,
                scrolling=False,
            )
            st.write(f"Phản hồi trong {processing_time:.2f} giây")