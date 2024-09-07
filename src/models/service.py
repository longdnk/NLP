from gemma.config import get_model_config
from gemma.model import GemmaForCausalLM
from datetime import datetime
import torch
# Choose variant and machine type
VARIANT = '2b-it' #@param ['2b', '2b-it', '9b', '9b-it', '27b', '27b-it']
MACHINE_TYPE = 'cpu' #@param ['cuda', 'cpu']

CONFIG = VARIANT[:2]
if CONFIG == '2b':
    CONFIG = '2b-v2'

item_dict = {
    'Tóm tắt ngắn gọn': 200, 
    'Tóm tắt theo ý chính': 1000,
    'Tóm tắt chi tiết': 1000,
}

class GemmaModel():
    def __init__(self):
        # Defaults Model
        self.weights_dir = "summarize/"
        self.tokenizer_path = self.weights_dir + 'tokenizer.model'
        self.checkpoint_path = self.weights_dir + 'model.ckpt'

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

        print(f"\033[93m[{formatted_time}]: Model predicting...\033[0m")
        # prompt_template = "<start_of_turn>Hãy tóm tắt đoạn văn sau và trích ra các sự kiện bên trong, bối cảnh và chi tiết nếu thấy cần thiết, câu trả lời được để sau ***Tóm tắt***: "
        # prompt_template = "<start_of_turn>Hãy tóm tắt đoạn văn sau, câu trả lời được để sau ***Tóm tắt***: "
        # input_sen = prompt_template + sentence + " <end_of_turn>"
        # input_sen.encode('utf-8')

        # prompt = f"<start_of_turn>Hãy tóm tắt đoạn văn sau, câu trả lời được để sau ***Tóm tắt***: {sentence} <end_of_turn>"
        
        prompt = ""
        if type == 'Tóm tắt ngắn gọn': 
            prompt = f"""<start_of_turn> 
                        Hãy tóm tắt đoạn văn sau và trả lời
                          {sentence} 
                        Lưu ý:
                            - Câu trả lời được để sau Tóm tắt:
                            - Chỉ tóm tắt ý, không thực hiện bất kỳ tác vụ nào khác.
                            - Luôn luôn trả lời bằng tiếng Việt.
                    <end_of_turn>"""
        elif type == 'Tóm tắt chi tiết': 
            prompt = f"""<start_of_turn> 
                        Hãy tóm tắt đoạn văn sau
                            {sentence} 
                        Lưu ý:
                            - Câu trả lời được để sau Tóm tắt:
                            - Tóm tắt một cách chi tiết nhất có thể.
                            - Độ dài khoảng 1000 từ.
                            - Có thể thêm các ý chính hoặc thông tin quan trọng.
                            - Nếu có các sự kiện hoặc cột mốc lịch sử trong văn bản, hãy trích thêm và nhấn mạnh những sự kiện này.
                            - Luôn luôn trả lời bằng tiếng Việt.
                        <end_of_turn>"""
        print(f"\033[95m[Prompt]: {prompt}\033[0m")

        # input_sen.encode('utf-8')
        prompt.encode('utf-8')

        # Generate sample
        results = self.model.generate(
            prompts=prompt,
            device=self.device,
            output_len=item_dict[type]
        )

        results = results.replace('<end_of_turn>', '') if len(results) > 0 else "Lỗi hệ thống, vui lòng thử lại."

        # Lấy thời gian hiện tại
        now = datetime.now()
        # Định dạng thời gian theo ngày/tháng/năm và giờ/phút/giây
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"\033[92mResult: {results}\033[0m")
        print(f"\033[93m[{formatted_time}]: End predicting...\033[0m")
        return results