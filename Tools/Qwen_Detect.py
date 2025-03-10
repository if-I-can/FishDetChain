# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch
# import json
# from PIL import Image, ImageDraw, ImageFont,ImageColor
# import xml.etree.ElementTree as ET
# import random
# import io
# import ast
# import xml.etree.ElementTree as ET
# import os


# def parse_json(json_output):
#     # Parsing out the markdown fencing
#     lines = json_output.splitlines()
#     for i, line in enumerate(lines):
#         if line == "```json":
#             json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
#             json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
#             break  # Exit the loop once "```json" is found
#     return json_output
# def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
#     """
#     Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

#     Args:
#         im: The image object.
#         bounding_boxes: A list of bounding boxes containing the name of the object
#          and their positions in normalized [y1 x1 y2 x2] format.
#         input_width: The width of the input image.
#         input_height: The height of the input image.
#     """

#     # Load the image
#     img = im
#     width, height = img.size
#     print(img.size)
#     # Create a drawing object
#     draw = ImageDraw.Draw(img)

#     # Define a list of colors
#     colors = [
#         'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray', 'beige',
#         'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
#         'lavender', 'violet', 'gold', 'silver'
#     ]

#     # Parsing out the markdown fencing
#     bounding_boxes = parse_json(bounding_boxes)

#     font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

#     try:
#         json_output = ast.literal_eval(bounding_boxes)
#     except Exception as e:
#         end_idx = bounding_boxes.rfind('"}') + len('"}')
#         truncated_text = bounding_boxes[:end_idx] + "]"
#         json_output = ast.literal_eval(truncated_text)

#     # Iterate over the bounding boxes
#     for i, bounding_box in enumerate(json_output):
#         # Select a color from the list
#         color = colors[i % len(colors)]

#         # Convert normalized coordinates to absolute coordinates
#         abs_y1 = int(bounding_box["bbox_2d"][1] / input_height * height)
#         abs_x1 = int(bounding_box["bbox_2d"][0] / input_width * width)
#         abs_y2 = int(bounding_box["bbox_2d"][3] / input_height * height)
#         abs_x2 = int(bounding_box["bbox_2d"][2] / input_width * width)

#         if abs_x1 > abs_x2:
#             abs_x1, abs_x2 = abs_x2, abs_x1

#         if abs_y1 > abs_y2:
#             abs_y1, abs_y2 = abs_y2, abs_y1

#         # Draw the bounding box
#         draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

#         # Draw the text
#         if "label" in bounding_box:
#             draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

#     # Ensure the directory exists and is writable
#     output_dir = '/home/zsl/DetToolChain/runs'
#     output_path = os.path.join(output_dir, 'ak.png')

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     if not os.access(output_dir, os.W_OK):
#         raise PermissionError(f"Cannot write to directory: {output_dir}")

#     # Save the image
#     img.save(output_path)
    
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         "Qwen/Qwen2.5-VL-7B-Instruct",
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#         device_map="auto",
#     )

# class Qwen_Detect:
#     def __init__(self):
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#                      "Qwen/Qwen2.5-VL-7B-Instruct",
#                     torch_dtype=torch.bfloat16,
#                     attn_implementation="flash_attention_2",
#                     device_map="auto",
#                     )
        

#     def message_process(self, boxes,input_image):
#         # Default processor
#         self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "image": input_image,
#                     },
#                     {
#                         "type": "text", 
#                         "text": f'基于Detect模型的检测框结果{boxes},请对这些检测框的位置信息进行微调，以确保检测准确，返回检测框的坐标，坐标格式为：左上角横坐标，左上角纵坐标，右下角横坐标，右下角纵坐标，格式为json'},
#                 ],
#             }
#         ]
        
#         text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(messages)

#         inputs = self.processor(text=[text],
#                                images=image_inputs,
#                                videos=video_inputs,
#                                padding=True,
#                                return_tensors="pt").to("cuda")
#         self.inputs = inputs

#     def result(self, input):
#         input_dict = json.loads(input)
#         print(input_dict)
#         print(input_dict)
#         boxes = input_dict.get('detect_yolo_results')
#         input_image = input_dict.get('image_path')
#         self.message_process(boxes,input_image)
#         generated_ids = self.model.generate(**self.inputs, max_new_tokens=512)
#         generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(self.inputs.input_ids, generated_ids)]
#         output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         return output_text

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
副本
"""""""""
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from PIL import Image, ImageDraw, ImageFont,ImageColor
import xml.etree.ElementTree as ET
import random
import io
import ast
import xml.etree.ElementTree as ET
import os


def parse_json(json_output):
    # 如果 json_output 是列表，将其转换为字符串
    if isinstance(json_output, list):
        json_output = "\n".join(json_output)  # 将列表中的元素用换行符连接成字符串

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
    img = Image.open(im)
    # Load the image

    width, height = img.size
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray', 'beige',
        'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
        'lavender', 'violet', 'gold', 'silver'
    ]

    # Parsing out the markdown fencing
    print(bounding_boxes)
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
    
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

class Qwen_Detect:
    def __init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                     "Qwen/Qwen2.5-VL-7B-Instruct",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                    )
        

    def message_process(self, boxes,input_image):
        # Default processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        print(boxes)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": input_image,
                    },
                    {
                        "type": "text", 
                        "text": f'这是所有鱼的位置信息{boxes},，请对位置信息进行调整，以提升检测精度，最后返回json格式的位置信息'},
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
    
    def result(self, input):
        input_dict = json.loads(input)
        boxes = input_dict.get('detect_yolo_results')
        input_image = input_dict.get('image_path')
        self.message_process(boxes,input_image)
        generated_ids = self.model.generate(**self.inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(self.inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        img = Image.open(input_image)
        width, height = img.size
        plot_bounding_boxes(input_image, output_text, width, height)
        return output_text
    


