from openai import OpenAI
import os
import base64
from PIL import Image
import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
#  base 64 编码格式

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# @title inference function with API

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
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

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
      print(bounding_box)
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      # abs_x1 = bounding_box["bbox_2d"][0]
      # abs_y1 = bounding_box["bbox_2d"][1]
      # abs_x2 = bounding_box["bbox_2d"][2]
      # abs_y2 = bounding_box["bbox_2d"][3]

      print(abs_x1,abs_y1,abs_x2,abs_y2)
      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1
      print(abs_x1,abs_y1,abs_x2,abs_y2)
      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Ensure the directory exists and is writable
      output_dir = '/home/zsl/FishDetChain/runs'
      output_path = os.path.join(output_dir, 'ak8.png')

      if not os.path.exists(output_dir):
          os.makedirs(output_dir, exist_ok=True)

      if not os.access(output_dir, os.W_OK):
          raise PermissionError(f"Cannot write to directory: {output_dir}")

    # Save the image
      img.save(output_path)


def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
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

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("output:\n",output_text[0])

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0], input_height, input_width


image_path = "/home/zsl/FishDetChain/dataset-example/01.jpg"


## Use a local HuggingFace model to inference.
# prompt in chinese
prompt = "框出每一个小鱼的位置，以json格式输出所有的坐标"
# prompt in english
# prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
response, input_height, input_width = inference(image_path, prompt)
image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)
