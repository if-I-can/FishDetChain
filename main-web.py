import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

class Interface:
    def __init__(self, model, input_text, input_image, tokens):
        self.model = model
        self.input_text = input_text
        self.input_image = input_image
        self.tokens = tokens

    def message_process(self):
        # Default processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.input_image,
                    },
                    {"type": "text", "text": self.input_text},
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

    def result(self):
        self.message_process()
        generated_ids = self.model.generate(**self.inputs, max_new_tokens=self.tokens)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(self.inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

def gradio_fn(input_text, input_image):
    interface_instance = Interface(model=model, input_text=input_text, input_image=input_image, tokens=5012)
    return interface_instance.result()

if __name__ == "__main__":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )


    demo = gr.Interface(
        fn=gradio_fn,  # Use the function gradio_fn to call the interface
        inputs=[
            gr.Textbox(label="Input Text"), 
            gr.Image(type="pil", label="Input Image"), 
        ],
        outputs="text",  # Output as text
    )

    demo.launch()
