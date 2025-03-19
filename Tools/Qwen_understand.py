from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch




class Interface:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

    def message_process(self, input_image):
        # Default processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                    "type": "image","image": input_image,
                    },
                    {"type": "text", "text": '帮我描述这张图像'},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(text=[text],
                               images=image_inputs,
                               videos=video_inputs,
                               padding=True,
                               return_tensors="pt").to("cuda")
        self.inputs = inputs

    def result(self, input_image):
        self.message_process(input_image)
        generated_ids = self.model.generate(**self.inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(self.inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

