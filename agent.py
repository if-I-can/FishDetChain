from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from select_tool import tools,tool_names


template = '''you are an intelligent agent named FishDetChain. Your final goal is to locate the target object in the image that described by given sentence, You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

The tool invocation must adhere to the following rules:

Follow these steps precisely:
First Tool (Mandatory): Use the image understanding tool to extract detailed information about the image content(Invoke only once). This step is essential to gain insights into the image before proceeding further.
Second Tool (Optional): Based on the content of the image, select and apply any appropriate image enhancement tool(You can invoke up to three tools, and each tool can be invoked at most once) to improve the quality or clarity of the image. Save the processed image for the next step.
Third Tool (Mandatory): Use the object detection tool on the enhanced image to identify and locate fish or other relevant objects(Invoke only once). This will generate initial detection results.
Fourth Tool (Mandatory): Input the detection results from the YOLO model into a vision large model for refinement. The large model should use its understanding of object shapes, sizes, 
and relationships in the image to fine-tune and adjust the bounding boxes. The output should be a refined JSON file containing the precise coordinates and bounding boxes of the detected objects, with a higher degree of accuracy and reduced errors and without reducing the number of detections.

Key Points:

Always start with the image understanding tool to analyze the image content.
Use the image enhancement tool only if necessary, based on the initial analysis.
Ensure the final output is a JSON file with accurate bounding box locations for the detected objects.
Example Workflow:
Input Image: /path/to/fish_image.jpg
Step 1: Understand the image content using the image understanding tool.
Step 2: Enhance the image (if needed) using the image enhancement tool.
Step 3: Detect objects in the image using the object detection tool.
Step 4: Fine-tune the detection results using the vision large model and output the JSON file,
Goal: Provide a clear, step-by-step reasoning chain to analyze fish images and produce accurate detection results in JSON format.


Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

class Agent():
    def __init__(self, api_key, api_url, system_prompt, model_name):

        # Setup the model
        self.Deepseek_model_config = ChatOpenAI(
            model=model_name,  # Model selection (e.g., gpt-3.5-turbo or gpt-4)
            openai_api_key=api_key,  # OpenAI API key
            openai_api_base=api_url,
            temperature=0.7,  # Control randomness of the generated text (0-1)
            max_tokens=None,  # Limit max tokens in the generated text
            n=1,  # Number of answers to return
        )

        # PromptTemplate with required variables


        # Use a simpler output parser
        self.output_parser = StrOutputParser()

        # Create the React agent without tools
        self.agent = create_react_agent(
            llm=self.Deepseek_model_config,
            tools=tools,  
            prompt=system_prompt,

        )

        # Agent executor
        self.agent_executor = AgentExecutor(agent=self.agent,tools=tools,verbose=True)

    def invoke(self, input):
        # Prepare the input with the required variables
        input_data = { "input": input  # Pass the user's input directly
        }
        return self.agent_executor.invoke(input_data)


if __name__ == '__main__':
    agent1 = Agent(
        api_key='sk-3bd0772e581b407d8e7601e75e150396', 
        api_url='https://api.deepseek.com',
        system_prompt=prompt,
        model_name='deepseek-chat'
    )

    # agent2 = Agent(
    #     api_key='sk-f8c6a280daa5450e9a6029bfc3d93553', 
    #     api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    #     system_prompt=prompt,
    #     model_name='deepseek-r1'
    # )

    print(agent1.invoke('请帮我框出图像中的每一条鱼，确保每条鱼是一个检测框，并以json格式返回检测到的每条鱼的位置信息，图像:/home/zsl/FishDetChain/dataset-example/01.jpg'))