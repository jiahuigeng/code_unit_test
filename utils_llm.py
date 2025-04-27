# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
# from transformers import AutoModel, AutoTokenizer
from api_resource import *
import torch
import torchvision.transforms as T
from openai import OpenAI
from PIL import Image
from io import BytesIO
import requests
import base64
import anthropic
import google.generativeai as genai
import mimetypes
from PIL import Image
import base64
import io
import time
import os
from typing import List, Dict, Any, Union, Optional
import base64

def load_image(image_path):
    """Load an image from the given path and return it as a base64-encoded string."""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_gpt_model(model_name):
    model = OpenAI(api_key=openai_api)
    return NamedModel(model, model_name)

# def prompt_gpt4o(model, prompt, image_path=None):
#     """
#     Generates a response from the GPT-4o model using optional image input.
#
#     Args:
#         model: The GPT-4o model instance.
#         prompt (str): The text prompt for the model.
#         image_path (str, optional): The file path of the image. Defaults to None.
#
#     Returns:
#         str: The model's response text.
#     """
#     try:
#         messages = [
#             {"role": "user", "content": [{"type": "text", "text": prompt}]}
#         ]
#
#         if image_path:
#             # Process the image into Base64 format
#             with open(image_path, "rb") as image_file:
#                 base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
#
#             # Append image information to the message content
#             messages[0]["content"].append(
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": base64_image},
#                 }
#             )
#
#         # Generate response from the model
#         response = model.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=messages,
#             temperature=0.01,
#             max_tokens=4096,
#         )
#
#         return response.choices[0].message.content
#
#     except Exception as e:
#         raise RuntimeError(f"An error occurred while querying the GPT-4o model: {e}")


import base64

def prompt_gpt4o(model, prompt, image_path=None, demonstrations=None):
    """
    Generates a response from the GPT-4o model using optional image input,
    and supports adding demonstration examples.

    Args:
        model: The GPT-4o model instance.
        prompt (str): The text prompt for the model.
        image_path (str, optional): The file path of the image. Defaults to None.
        demonstrations (list of tuples, optional):
            A list of (demo_query, demo_image_path, demo_answer).
            Each tuple is used as a demonstration example before the final prompt.

    Returns:
        str: The model's response text.
    """
    try:
        messages = []

        # 1. Add demonstration pairs, if any
        if demonstrations:
            for (demo_query, demo_image_path, demo_answer) in demonstrations:
                # (A) User demonstration message
                user_content = [{"type": "text", "text": demo_query}]

                if demo_image_path:
                    # Convert demo image to base64
                    with open(demo_image_path, "rb") as image_file:
                        base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_image},
                        }
                    )

                messages.append({"role": "user", "content": user_content})

                # (B) Assistant demonstration message (the "answer")
                assistant_content = [{"type": "text", "text": demo_answer}]
                messages.append({"role": "assistant", "content": assistant_content})

        # 2. Add the final user prompt
        final_user_content = [{"type": "text", "text": prompt}]

        if image_path:
            # Convert final prompt image to base64
            with open(image_path, "rb") as image_file:
                base64_image = f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
            final_user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                }
            )

        messages.append({"role": "user", "content": final_user_content})

        # 3. Generate response from the model
        response = model.model.chat.completions.create(
            model=model.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the GPT-4o model: {e}")


class NamedModel:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

def get_gemini_model(model_name='gemini-2.0-flash'):
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel(model_name)
    return NamedModel(model, model_name)

# def get_gemini_model():
#     genai.configure(api_key=gemini_api)
#     client = genai.GenerativeModel('gemini-1.5-flash')
#     return client

def get_claude(model_name="claude-3-haiku") -> anthropic.Anthropic:
    """
    创建并返回一个Claude API客户端实例。

    参数:
        api_key (str): Anthropic API密钥

    返回:
        anthropic.Anthropic: Claude API客户端实例
    """
    model = anthropic.Anthropic(api_key=anthropic_api_key)
    return NamedModel(model, model_name)
def prompt_gemini(client, prompt, image_path=None, demonstrations=None):
    """
    Prompt Gemini with optional in-context learning using demonstrations.

    Args:
        client: The initialized Gemini model client
        prompt (str): The text prompt to send to Gemini
        image_path (str, optional): Path to an image file to include in the prompt
        demonstrations (list, optional): List of tuples (prompt, image_path, answer) for in-context learning

    Returns:
        str: Gemini's response
    """
    import PIL.Image

    # Initialize the content list that will be sent to Gemini
    content_parts = []

    # Add demonstrations for in-context learning if provided
    if demonstrations and len(demonstrations) > 0:
        # Format the demonstrations
        for demo_prompt, demo_image_path, demo_answer in demonstrations:
            # Add a separator for each demonstration
            content_parts.append("Example Input:")

            # Add demonstration image if provided
            if demo_image_path:
                try:
                    demo_image = PIL.Image.open(demo_image_path)
                    content_parts.append(demo_image)
                except Exception as e:
                    content_parts.append(f"[Image could not be loaded: {str(e)}]")

            # Add demonstration prompt
            content_parts.append(demo_prompt)

            # Add demonstration answer
            content_parts.append("Example Output:")
            content_parts.append(demo_answer)

            # Add a separator between demonstrations
            content_parts.append("---")

        # Add a final separator before the actual query
        content_parts.append("Now, please analyze the following:")

    # Add the current image if provided
    if image_path:
        try:
            image = PIL.Image.open(image_path)
            content_parts.append(image)
        except Exception as e:
            return f"Error loading image: {str(e)}"

    # Add the current prompt
    content_parts.append(prompt)

    # Generate the response
    try:
        response = client.model.generate_content(content_parts, generation_config={"temperature": 0.0})
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


def prompt_claude(client,
                  prompt: str,
                  image_path: Optional[str] = None,
                  demonstrations: Optional[List[Dict[str, str]]] = None,
                  model: str = "claude-3-haiku-20240307") -> str:
    """
    向Claude发送提示并获取回复。

    参数:
        client (anthropic.Anthropic): Claude API客户端实例
        prompt (str): 发送给Claude的提示文本
        image_path (str, optional): 可选的图片文件路径，用于多模态输入
        demonstrations (List[Dict[str, str]], optional): 用于少样本学习的示例列表
        model (str): 要使用的Claude模型，默认为"claude-3-haiku-20240307"

    返回:
        str: Claude的回复文本
    """
    messages = []

    # 添加少样本学习的示例（如果提供）
    if demonstrations:
        for demo in demonstrations:
            if "user" in demo:
                messages.append({"role": "user", "content": demo["user"]})
            if "assistant" in demo:
                messages.append({"role": "assistant", "content": demo["assistant"]})

    # 准备用户消息内容
    content = []

    # 添加文本提示
    content.append({"type": "text", "text": prompt})

    # 如果提供了图片，添加图片
    if image_path:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": f"image/{image_path.split('.')[-1]}",
                "data": base64_image
            }
        })

    # 添加用户消息
    messages.append({"role": "user", "content": content})

    # 调用API获取回复
    response = client.model.messages.create(
        model=model,
        messages=messages,
        max_tokens=4096
    )

    # 返回回复文本
    return response.content[0].text

# def prompt_gemini(client, prompt, image_path=None, demonstrations=None):
#     """
#     Generates a response from the Gemini model using optional `image input and few-shot demonstrations.
#
#     Args:
#         client: The Gemini model client.
#         prompt (str): The text prompt for the model.
#         image_path (str, optional): The path to the image to load. Defaults to None.
#         demonstrations (list, optional): A list of few-shot demonstrations. Defaults to None.
#
#     Returns:
#         str: The model's response text.
#     """
#     try:
#         # Prepare the inputs as a list of parts
#         parts = []
#
#         # Add demonstrations as context
#         if demonstrations:
#             for (demo_prompt, demo_image_path, demo_answer) in demonstrations:
#                 # Add demo image if image path is provided
#                 if demo_image_path:
#                     demo_image_data = load_image(demo_image_path)
#                     parts.append({
#                         "inline_data": {
#                             "mime_type": "image/jpeg",  # Adjust mime type if necessary
#                             "data": demo_image_data
#                         }
#                     })
#
#                 # Add demo prompt
#                 parts.append({"text": demo_prompt})
#
#                 # Add demo answer
#                 parts.append({"text": demo_answer})
#
#         # Add the actual user prompt and image
#         if image_path:
#             image_data = load_image(image_path)
#             parts.append({
#                 "inline_data": {
#                     "mime_type": "image/jpeg",  # Adjust mime type if necessary
#                     "data": image_data
#                 }
#             })
#
#         # Add the user prompt
#         parts.append({"text": prompt})
#
#         # Generate content using the provided parts
#         response = client.generate_content({"parts": parts})
#         return response.text
#
#     except Exception as e:
#         raise RuntimeError(f"An error occurred while querying the Gemini model: {e}")


def get_commercial_model(model_name):
    model_map = {
        "gpt-4o": get_gpt_model,
        "gpt-4o-mini": get_gpt_model,
        "gemini-2.0-flash": get_gemini_model,
        "claude-3-haiku": get_claude,
    }


    if model_name in model_map:
        return model_map[model_name](model_name)
    raise ValueError(f"Unknown model name: {model_name}")

def prompt_commercial_model(client, model_name, prompt, image_id, demonstrations=None):
    # print(prompt)
    prompt_map = {
        "gpt-4o": prompt_gpt4o,
        "gpt-4o-mini": prompt_gpt4o,
        "gemini-2.0-flash": prompt_gemini,
        "claude-3-haiku": prompt_claude,
    }
    if model_name in prompt_map:
        try:
            res = prompt_map[model_name](client, prompt, image_id, demonstrations)
            print(res)
            return res
        except Exception as e:
            print(image_id, str(e))
            return ""
    raise ValueError(f"Unknown model name: {model_name}")


if __name__ == "__main__":
    gpt_client = get_gpt_model("gpt-4o-mini")
    gemini_client = get_gemini_model()
    claude_client = get_claude()
    # image_path = "demo.jpeg"
    # text_prompt = "describe the image"
    text_prompt = "where is the capital of China?"

    # # Process and get response
    # response = prompt_gpt4o(gpt_client, text_prompt, "",)

    # Display the response
    # print("\ngpt-4o Response:")
    # print(response)
    # Process and get response
    # response = prompt_gemini(gemini_client, text_prompt, "",)
    #
    # # Display the response
    # print("\nGemini Response:")
    # print(response)

    # Display the response
    # response = prompt_gemini(claude_client, text_prompt, "", )
    # print("\nClaude Response:")
    # print(response)

    response = prompt_claude(claude_client, text_prompt, "")
    print("\nClaude Response:")
    print(response)
