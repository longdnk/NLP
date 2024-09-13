import torch
import helper.prompt as prompt_generator
from gemma.config import get_model_config
from gemma.model import GemmaForCausalLM
from datetime import datetime

# Choose variant and machine type
VARIANT = "2b-it"
MACHINE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = VARIANT[:2]
if CONFIG == "2b":
    CONFIG = "2b-v2"

print(f"\033[95mInit model 2B success\033[0m")


class Model2B:
    def __init__(self):
        # Xoá bộ đệm
        torch.cuda.empty_cache()
        # Defaults Model
        self.weights_dir = "weights/"
        self.tokenizer_path = self.weights_dir + "tokenizer-2b.model"
        self.checkpoint_path = self.weights_dir + "model-2b.ckpt"

        # Set up model config.
        model_config = get_model_config(CONFIG)
        model_config.tokenizer = self.tokenizer_path
        model_config.quant = "quant" in VARIANT

        # Instantiate the model and load the weights.
        torch.set_default_dtype(model_config.get_dtype())
        self.device = torch.device(MACHINE_TYPE)
        model = GemmaForCausalLM(model_config)
        model.load_weights(self.checkpoint_path)
        self.model = model.to(self.device).eval()

    def predict(self, sentence: str, type: str, compression: float):
        # Lấy thời gian hiện tại
        now = datetime.now()
        # Định dạng thời gian theo ngày/tháng/năm và giờ/phút/giây
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")

        word_count = len(sentence.split())
        
        limit = 0 

        print(f"\033[93m[{formatted_time}]: Model 2b predicting...\033[0m")

        prompt = ""
        if type == "Tóm tắt ngắn gọn":
            limit = int(word_count * compression)
            prompt = prompt_generator.short_predict_prompt(
                sentence=sentence, limit=limit + 1
            )

        elif type == "Tóm tắt chi tiết":
            limit = int(word_count * compression)
            prompt = prompt_generator.long_predict_prompt(
                sentence=sentence, limit=limit + 1
            )

        print(f"\033[95m[Prompt]: {prompt}\033[0m")

        prompt.encode("utf-8")

        # Generate sample
        result = self.model.generate(
            prompts=prompt, device=self.device, output_len=limit + 1
        )

        result = result.replace("<end_of_turn>", "")
        result = result.replace("<div>", "")
        result = result.replace("</div>", "")
        result = result if len(result) > 0 else "Lỗi hệ thống, vui lòng thử lại."
        # Lấy thời gian hiện tại
        now = datetime.now()
        # Định dạng thời gian theo ngày/tháng/năm và giờ/phút/giây
        formatted_time = now.strftime("%d/%m/%Y %H:%M:%S")
        print(f"\033[92mResult: {result}\033[0m")
        print(f"\033[93m[{formatted_time}]: End predicting...\033[0m")
        return result
