from gemma.config import get_model_config
from gemma.model import GemmaForCausalLM
from datetime import datetime
import torch
# Choose variant and machine type
VARIANT = '2b-it'
MACHINE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = VARIANT[:2]
if CONFIG == '2b':
    CONFIG = '2b-v2'

class SummarizeModel():
    def __init__(self):
        # Defaults Model
        self.weights_dir = "weights/"
        self.tokenizer_path = self.weights_dir + 'tokenizer-2b.model'
        self.checkpoint_path = self.weights_dir + 'model-2b.ckpt'

        # Set up model config.
        model_config = get_model_config(CONFIG)
        model_config.tokenizer = self.tokenizer_path
        model_config.quant = 'quant' in VARIANT

        # Instantiate the model and load the weights.
        torch.set_default_dtype(model_config.get_dtype())
        self.device = torch.device(MACHINE_TYPE)
        model = GemmaForCausalLM(model_config)
        model.load_weights(self.checkpoint_path)
        self.model = model.to(self.device).eval()

    def predict(self, sentence: str, type: str):
        # Lấy thời gian hiện tại
        now = datetime.now()
        # Định dạng thời gian theo ngày/tháng/năm và giờ/phút/giây
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")

        word_count = len(sentence.split(' '))
        limit = 200

        print(f"\033[93m[{formatted_time}]: Model predicting...\033[0m")

        prompt = ""
        if type == 'Tóm tắt ngắn gọn': 
            low_range = int(word_count * 0.5)
            limit = low_range if low_range > limit else limit
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
        elif type == 'Tóm tắt chi tiết': 
            limit = int(word_count * 0.8)
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
                            - Có thể thêm các ý chính hoặc thông tin quan trọng.
                            - Độ dài tối đa {limit} từ.
                            - Nếu có các sự kiện hoặc cột mốc lịch sử trong văn bản, 
                              hãy trích thêm và nhấn mạnh những sự kiện này.
                        <end_of_turn>
                        """
        print(f"\033[95m[Prompt]: {prompt}\033[0m")

        prompt.encode('utf-8')

        # Generate sample
        result = self.model.generate(
            prompts=prompt,
            device=self.device,
            output_len=limit
        )

        result = result.replace('<end_of_turn>', '')
        result = result.replace('<div>', '')
        result = result.replace('</div>', '')
        result = result if len(result) > 0 else "Lỗi hệ thống, vui lòng thử lại."

        # Lấy thời gian hiện tại
        now = datetime.now()
        # Định dạng thời gian theo ngày/tháng/năm và giờ/phút/giây
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"\033[92mResult: {result}\033[0m")
        print(f"\033[93m[{formatted_time}]: End predicting...\033[0m")
        return result
