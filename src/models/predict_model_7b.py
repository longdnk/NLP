import torch
import helper.prompt as prompt_generator
from gemma.config import get_model_config
from gemma.model import GemmaForCausalLM
from datetime import datetime
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'
# Choose variant and machine type
MACHINE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'

VARIANT = '7b-it-quant'
CONFIG = '7b'

print(f"\033[95mInit model 7B success\033[0m")

class Model7B():
    def __init__(self):
        # Xoá bộ đệm
        torch.cuda.empty_cache()
        # Defaults Model
        self.weights_dir = "weights/"
        self.tokenizer_path = self.weights_dir + 'tokenizer-7b.model'
        self.checkpoint_path = self.weights_dir + 'model-7b-quantization.ckpt'

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

        print(f"\033[93m[{formatted_time}]: Model 7b predicting...\033[0m")

        prompt = ""
        if type == 'Tóm tắt ngắn gọn': 
            low_range = int(word_count * 0.5)
            limit = low_range if low_range > limit else limit
            prompt = prompt_generator.short_predict_prompt(sentence=sentence, limit=limit)

        elif type == 'Tóm tắt chi tiết': 
            limit = int(word_count * 0.8)
            prompt = prompt_generator.long_predict_prompt(sentence=sentence, limit=limit)

        print(f"\033[95m[Prompt]: {prompt}\033[0m")

        prompt.encode('utf-8')

        result = "OK DONE"
        # Generate sample
        # result = self.model.generate(
        #     prompts=prompt,
        #     device=self.device,
        #     output_len=limit
        # )

        # result = result.replace('<end_of_turn>', '')
        # result = result.replace('<div>', '')
        # result = result.replace('</div>', '')
        # result = result if len(result) > 0 else "Lỗi hệ thống, vui lòng thử lại."

        # Lấy thời gian hiện tại
        now = datetime.now()
        # Định dạng thời gian theo ngày/tháng/năm và giờ/phút/giây
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"\033[92mResult: {result}\033[0m")
        print(f"\033[93m[{formatted_time}]: End predicting...\033[0m")
        return result
