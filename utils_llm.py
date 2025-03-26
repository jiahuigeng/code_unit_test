from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
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

def load_image(image_path):
    """Load an image from the given path and return it as a base64-encoded string."""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format=img.format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_gpt_model():
    model = OpenAI(api_key=openai_api)
    return model

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
        response = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            max_tokens=4096,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying the GPT-4o model: {e}")



def get_gemini_model():
    genai.configure(api_key=gemini_api)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

# def get_gemini_model():
#     genai.configure(api_key=gemini_api)
#     client = genai.GenerativeModel('gemini-1.5-flash')
#     return client

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
        response = client.generate_content(content_parts, generation_config={"temperature": 0.0})
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# def prompt_gemini(client, prompt, image_path=None, demonstrations=None):
#     """
#     Generates a response from the Gemini model using optional image input and few-shot demonstrations.
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
        "gpt4o": get_gpt_model,
        "gemini": get_gemini_model,
    }
    if model_name in model_map:
        return model_map[model_name]()
    raise ValueError(f"Unknown model name: {model_name}")

def prompt_commercial_model(client, model_name, prompt, image_id, demonstrations=None):
    # print(prompt)
    prompt_map = {
        "gpt4o": prompt_gpt4o,
        "gemini": prompt_gemini,
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
    gpt_client = get_gpt_model()
    gemini_client = get_gemini_model()
    # image_path = "demo.jpeg"
    # text_prompt = "describe the image"
    text_prompt = "where is the capital of China?"

    # Process and get response
    response = prompt_gpt4o(gpt_client, text_prompt, "",)

    # Display the response
    print("\ngpt-4o Response:")
    print(response)

    # Process and get response
    response = prompt_gemini(gemini_client, text_prompt, "",)

    # Display the response
    print("\nGemini Response:")
    print(response)
