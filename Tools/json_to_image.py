from PIL import Image, ImageDraw, ImageFont,ImageColor
import xml.etree.ElementTree as ET
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import random
import io
import ast
import xml.etree.ElementTree as ET
import os

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

model_path = "/home/zsl/DetToolChain/Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        im: The image object.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
        input_width: The width of the input image.
        input_height: The height of the input image.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray', 'beige',
        'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
        'lavender', 'violet', 'gold', 'silver'
    ]

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1] / input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Ensure the directory exists and is writable
    output_dir = '/home/zsl/DetToolChain/runs'
    output_path = os.path.join(output_dir, 'ak.png')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if not os.access(output_dir, os.W_OK):
        raise PermissionError(f"Cannot write to directory: {output_dir}")

    # Save the image
    img.save(output_path)


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image

class QwenVLInference:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/home/zsl/DetToolChain/Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    def inference(self, img_path, prompt, max_new_tokens=10240):
        image = Image.open(img_path)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        input_height = inputs['image_grid_thw'][0][1] * 14
        input_width = inputs['image_grid_thw'][0][2] * 14
        
        return output_text[0], input_height, input_width

    def plot_results(self, img_path, response, input_width, input_height):
        image = Image.open(img_path)
        print("Original Image Size:", image.size)
        image.thumbnail([640, 640], Image.Resampling.LANCZOS)
        plot_bounding_boxes(image, response, input_width, input_height)

# Example usage
if __name__ == "__main__":
    model = QwenVLInference()
    image_path = "/home/zsl/DetToolChain/dataset-example/01.jpg"
    prompt = "框出每一条鱼的位置，以JSON格式输出所有的坐标"
    response, input_height, input_width = model.inference(image_path, prompt)
    print(response)
    model.plot_results(image_path, response, input_width, input_height)

