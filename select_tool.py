from Tools import *
from langchain.tools import Tool


"""
Tool1: Qwen_understand
first to understand the image and return the result
"""

Qwen_understand_initial = Interface()
Qwen_detect_initial = Qwen_Detect()

def Qwen_understand_result(input_image):
    Qwen_understand_result = Qwen_understand_initial.result(input_image)
    return Qwen_understand_result

def Qwen_Detect_result(input):
    Qwen_detect_result = Qwen_detect_initial.result(input)
    return Qwen_detect_result



tools = [
    Tool(
        name="understand image",
        func=Qwen_understand_result,
        description="调用视觉大模型以返回图像描述,输入为图像路径",
    ),
    Tool(
        name="detect_yolo",
        func=detect_yolo,
        description="该工具为YOLO检测模型,用以获取输入图像中物体的类别和位置信息",
    ),
    Tool(
        name="Qwen_Detect_result",
        func=Qwen_Detect_result,
        description="调用视觉大模型,对detect_yolo中输出物体的类别和位置信息进行微调,确保检测准确,输入为detect_yolo的检测结果和输入图像路径",
    ),
]


tool_names = [tool.name for tool in tools]




