"""
Detect certain object in the image
"""
image_path = "./assets/spatial_understanding/cakes.png"


## Use a local HuggingFace model to inference.
# prompt in chinese
prompt = "框出每一个小蛋糕的位置，以json格式输出所有的坐标"
# prompt in english
prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
print(image.size)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)

"""
 Detect a specific object using descriptions
"""
image_path = "./assets/spatial_understanding/cakes.png"

# prompt in chinses
prompt = "定位最右上角的棕色蛋糕，以JSON格式输出其bbox坐标"
# prompt in english
prompt = "Locate the top right brown cake, output its bbox coordinates using JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)

## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)

"""
Point to certain objects in xml format
"""
image_path = "./assets/spatial_understanding/cakes.png"

# prompt in chinese
prompt = "以点的形式定位图中桌子远处的擀面杖，以XML格式输出其坐标"
# prompt in english
prompt = "point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>"

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_points(image, response, input_width, input_height)

"""
Reasoning capability
"""
image_path = "./assets/spatial_understanding/Origamis.jpg"

# prompt in chinese
prompt = "框出图中纸狐狸的影子，以json格式输出其bbox坐标"
# prompt in english
prompt = "Locate the shadow of the paper fox, report the bbox coordinates in JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, response, input_width, input_height)

## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)

"""
Understand relationships across different instances
"""
image_path = "./assets/spatial_understanding/cartoon_brave_person.jpeg"

# prompt in chinese
prompt = "框出图中见义勇为的人，以json格式输出其bbox坐标"
# prompt in english
prompt = "Locate the person who act bravely, report the bbox coordinates in JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, response, input_width, input_height)


## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)
"""
Find a special instance with unique characteristic (color, location, utility, ...)
"""
url = "./assets/spatial_understanding/multiple_items.png"

# prompt in chinese
prompt = "如果太阳很刺眼，我应该用这张图中的什么物品，框出该物品在图中的bbox坐标，并以json格式输出"
# prompt in english
prompt = "If the sun is very glaring, which item in this image should I use? Please locate it in the image with its bbox coordinates and its name and output in JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(url, prompt)

image = Image.open(url)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, response, input_width, input_height)


## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)

"""
Use Qwen2.5-VL grounding capabilities to help counting
"""
image_path = "./assets/spatial_understanding/multiple_items.png"

# prompt in chinese
prompt = "请以JSON格式输出图中所有物体bbox的坐标以及它们的名字，然后基于检测结果回答以下问题：图中物体的数目是多少？"
# prompt in english
prompt = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)

# # Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)

"""
spatial understanding with designed system prompt
"""
image_path = "./assets/spatial_understanding/cakes.png"
image = Image.open(image_path)
system_prompt = "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
prompt = "find all cakes"
response, input_height, input_width = inference(image_path, prompt, system_prompt=system_prompt)
