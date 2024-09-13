import streamlit as st
import requests
from io import BytesIO
import docx2txt
import fitz  # PyMuPDF
import os

response_code = 200

# Set the app to wide mode
st.set_page_config(layout="wide")

st.title("Công cụ tóm tắt văn bản")

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
        "Nhập văn bản cần tóm tắt, tối đa 5000 từ",
        value=prompt,
        height=300,
        max_chars=5000,
        key="input_text_area",
    )

    # Add input for summary type selection below the text area
    summary_type = st.selectbox(
        "Chọn loại tóm tắt:", ("Tóm tắt ngắn gọn", "Tóm tắt chi tiết")
    )

    # Add input for compression ratio
    compression_ratio = st.selectbox(
        "Chọn tỷ lệ nén:",
        [50, 60, 70, 80, 90] if summary_type == "Tóm tắt chi tiết" else [50, 60, 70],
    )
    st.session_state.compression_ratio = compression_ratio

    # Add model selection
    model_type = st.selectbox("Chọn mô hình:", ("2b", "7b"))

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
            <p>5. Bấm nút Tóm Tắt và tận hưởng kết quả.</p>
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
        with st.spinner("Đang xử lý..."):
            try:
                # Send request to API
                api_url = (
                    "http://192.169.50.47:5005/predict"
                    if model_type == "2b"
                    else "http://192.168.50.110:5005/predict"
                )  # Replace with actual API URL

                headers = {
                    "Access-Control-Allow-Origin": "*",
                    "X-Requested-With": "XMLHttpRequest",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }

                data = {
                    "text": prompt,
                    "type": summary_type,
                    "model": model_type,
                    "compression": compression_ratio,
                }

                response = requests.post(api_url, json=data, headers=headers)

                # Handle API response
                if response.status_code == 200:
                    response_data = response.json()
                    result = response_data["data"]
                    st.success("Hoàn thành")  # Using st.success for success message
                # else:
                #     result = "Error: Unable to summarize the text."
            except Exception as e:
                response_code = 400
                result = f"Error: {str(e)}"

            # Store the result in session state
            st.session_state.result_text = result

    # Display result if available with max height of 400px and scroll enabled
    if st.session_state.result_text:
        st.markdown(
            f"""
                <div style="max-height: 400px; overflow-y: auto; padding: 10px;">
                    {st.session_state.result_text}</div>
            """,
            unsafe_allow_html=True,
        )
        # Copy to clipboard button with icon
        if response_code == 200 and st.button("Sao chép vào bộ nhớ đệm", key="copy_button"):
            st.success(
                "Đã sao chép vào bộ nhớ đệm!"
            )  # Using st.success for copy confirmation
