from tools import inference
from tools import plot_bounding_boxes
from PIL import Image
import dashscope
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
dashscope.api_key = "your_api_key"

def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=10240):

    image = Image.open(img_url)
    messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
  ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("input:\n",text)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=5012)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("output:\n",output_text[0])

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0], input_height, input_width


image_path = "/home/zsl/DetToolChain/2.png"
## Use a local HuggingFace model to inference.
# prompt in chinese
prompt = "请必须帮我框出每一小鱼的位置，尽可能标出所有的鱼，以json格式返回其位置信息，包括左上角和右下角的坐标"
# prompt in english
# prompt = "Outline the position of each small fish and output all the coordinates in JSON format."
# response, input_height, input_width = inference(image_path, prompt)
response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-72b-instruct', messages=messages)
image = Image.open(image_path)
# image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)