import streamlit as st
import requests
from io import BytesIO

# Set the app to wide mode
st.set_page_config(layout="wide")

st.title("Công cụ tóm tắt văn bản")

# Initialize session state for storing assistant messages and result text
if "assistant_messages" not in st.session_state:
    st.session_state.assistant_messages = []
if "result_text" not in st.session_state:
    st.session_state.result_text = ""

# Split the page into two columns
col1, col2 = st.columns(2)

# Left column: User input and summary type selection
with col1:
    # Initialize prompt variable
    prompt = ""

    # Add file uploader for text files
    uploaded_file = st.file_uploader("Tải lên một tệp văn bản (.txt)", type=["txt"])

    # If a file is uploaded, read its content and set it as the prompt
    if uploaded_file is not None:
        prompt = uploaded_file.read().decode("utf-8")

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
            <p>3. Bấm nút Tóm Tắt và tận hưởng kết quả.</p>
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

    # Display loading state
    result = ""
    if submit and prompt:
        with st.spinner("Đang xử lý..."):
            try:
                # Send request to API
                api_url = "http://192.168.50.198:5005/predict"  # Replace with actual API URL
                headers = {"Content-Type": "application/json"}
                data = {"text": prompt, "type": summary_type}

                response = requests.post(api_url, json=data, headers=headers)

                # Handle API response
                if response.status_code == 200:
                    response_data = response.json()
                    st.toast("Hoàn thành", icon="✅")
                    result = response_data["data"]
                else:
                    result = "Error: Unable to summarize the text."
            except Exception as e:
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
