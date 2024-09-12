def short_predict_prompt(sentence: str, limit: int):
    prompt = f"""<start_of_turn> 
                Hãy tóm tắt đoạn văn sau
                    {sentence} 
                Lưu ý:
                    - Văn bản đã tóm tắt phải là đơn văn bản.
                    - Văn bản đã tóm tắt phải được viết bằng ngôn ngữ tiếng Việt.
                    - Văn bản đã tóm tắt phải được tổ chức theo định dạng Markdown, trong đó phải có
                        cụm '## Tóm tắt: ' ở đầu văn bản, văn bản đã tóm tắt phải được đặt sau cụm này.
                    - Văn bản đã tóm tắt phải được chuẩn hoá, loại bỏ các khoảng trống dư thừa.
                    - Chỉ tóm tắt ý, không thực hiện bất kỳ tác vụ nào khác.
                    - Độ dài tối đa {limit} từ.
                <end_of_turn>
                """
    return prompt

def long_predict_prompt(sentence: str, limit: int):
    prompt = f"""<start_of_turn> 
                Hãy tóm tắt đoạn văn sau
                    {sentence} 
                Lưu ý:
                    - Văn bản đã tóm tắt phải là đơn văn bản.
                    - Văn bản đã tóm tắt phải được viết bằng ngôn ngữ tiếng Việt.
                    - Văn bản đã tóm tắt phải được tổ chức theo định dạng Markdown, trong đó phải có
                        cụm '## Tóm tắt: ' ở đầu văn bản, văn bản đã tóm tắt phải được đặt sau cụm này.
                    - Văn bản đã tóm tắt phải được chuẩn hoá, loại bỏ các khoảng trống dư thừa.
                    - Tóm tắt một cách chi tiết nhất có thể.
                    - Độ dài tối đa {limit} từ.
                    - Nếu có các sự kiện, ý chỉnh, thông tin quan trọng hoặc cột mốc lịch sử trong văn bản
                        thì bạn hãy đưa vào trong văn bản tóm tắt nhé.
                    - Vui lòng viết tất cả thành 1 đoạn văn chứ không liệt kê ý.
                <end_of_turn>
                """
    return prompt