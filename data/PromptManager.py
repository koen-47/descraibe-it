import itertools
import os
import re
import time

import openai
import pandas as pd
from tqdm import tqdm


class PromptManager:
    """
    Class to handle all functionality related to prompting ChatGPT for word descriptions.
    """
    def __init__(self, api_key, save_path, categories_file="./data/saved/categories_25.txt"):
        """
        Constructor for PromptManager class.
        :param api_key: OpenAI API key for ChatGPT
        :param args: Dictionary of arguments used for prompting (found in main.py)
        """
        self._api_key = api_key
        openai.key = api_key
        self._save_path = save_path
        self._categories_file = categories_file

        self._PROMPT_INFO = {
            "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                               '"<var4>" or any of its variations in your response. Use <var5> language in your '
                               'response.<var6>',
            "length": [20],
            "detail": ["very short", "short", "", "long", "very long"],
            "complexity": ["very simple", "simple", "complex", "very complex"],
            "prefix": ["it", "this", "a", "the", "with", ""],
            "temperature": [0.2, 0.6, 1.0],
        }

    def start_prompts(self):
        """
        Class function that designs all the prompts, sends them to the ChatGPT API, adds all the responses to a
        Pandas DataFrame, and saves the frame to a CSV file.
        """
        df = pd.DataFrame({"description": [], "label": []})
        categories = self.get_categories(self._categories_file)
        for category in categories:
            responses = []
            variables = [self._PROMPT_INFO["length"], self._PROMPT_INFO["detail"], self._PROMPT_INFO["complexity"],
                         self._PROMPT_INFO["prefix"], self._PROMPT_INFO["temperature"]]
            variations = list(itertools.product(*variables))
            for i in tqdm(range(len(variations)), desc=f"Prompting ({category})"):
                variation = variations[i]
                prompt = self.prepare_prompt(category, length=variation[0], detail=variation[1],
                                             complexity=variation[2], prefix=variation[3])
                response = self.make_safe_prompt(prompt, temperature=variation[4])
                response = self.__clean_responses(response.split("\n"))
                responses.append(response)

            for row in responses:
                row = pd.DataFrame({"description": row, "label": [category] * len(row)})
                df = pd.concat([df, row], ignore_index=True)
            df.to_csv(self._save_path, index=False)

    def prepare_prompt(self, entity, length=1, detail=None, complexity=None, prefix=None):
        """
        Designs a prompt to be sent to the ChatGPT API.
        :param entity: The word that will be described (string)
        :param length: The number of descriptions to be received from ChatGPT (integer)
        :param detail: The level of detail that should be in ChatGPT's response.
        :param complexity: The language complexity that should be ChatGPT's response.
        :param prefix: The word that ChatGPT should use to start its response.
        :return: Returns the formatted prompt as a string.
        """
        template = self._PROMPT_INFO["prompt_template"]
        article = "an" if entity[0] in ["a", "e", "i", "o", "u"] else "a"
        detail = self.__prepare_prompt_detail(detail)
        prefix = f' Start all your responses with "{prefix.capitalize()}".' if prefix != "" else ""
        template = re.sub(f"<var1>", str(length), template)
        template = re.sub(f"<var2>", detail, template)
        template = re.sub(f"<var3>", f"{article} {entity}", template)
        template = re.sub(f"<var4>", f"{entity}", template)
        template = re.sub(f"<var5>", complexity, template)
        template = re.sub(f"<var6>", prefix, template)
        return template

    def __prepare_prompt_detail(self, detail):
        if detail == "very short":
            return "very short and "
        if detail == "short":
            return "short and "
        if detail == "long":
            return "detailed and "
        if detail == "very long":
            return "very detailed and "
        return ""

    def __clean_responses(self, responses):
        """
        Cleans the response minimally so it can be saved to the CSV file.
        :param responses: The raw text response from ChatGPT.
        :return: Returns the cleaned response from ChatGPT.
        """
        cleaned = []
        for i in range(len(responses)):
            if responses[i] != "":
                cleaned.append(responses[i])
        return cleaned

    def make_safe_prompt(self, prompt, temperature=0.7, max_retries=30, timeout=30):
        """
        Performs a safe prompt to the ChatGPT API that catches the APIError and RateLimitError exception that retries
        a number of times in case of failure.
        :param prompt: The prompt to be sent to ChatGPT's API.
        :param temperature: Temperature variable of ChatGPT.
        :param max_retries: The maximum number of retries to perform in case of failure.
        :param timeout: The time between retries (in seconds)
        :return: Returns the raw response from ChatGPT.
        """
        try:
            return self.__make_prompt(prompt, temperature=temperature)
        except (openai.error.APIError, openai.error.RateLimitError, openai.error.Timeout):
            retries = 1
            while retries <= max_retries:
                print(f"\n{retries}. Error. Restarting. Prompt = {prompt}")
                try:
                    return self.__make_prompt(prompt, temperature=temperature)
                except (openai.error.APIError, openai.error.RateLimitError, openai.error.Timeout):
                    time.sleep(timeout)
                    retries += 1

    def __make_prompt(self, prompt, temperature=0.7):
        """
        Perform an unsafe prompt to ChatGPT's API.
        :param prompt: The prompt to be sent to ChatGPT.
        :param temperature: Temperature variable of ChatGPT.
        :return: The response sent back from ChatGPT.
        """
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": prompt}
            ],
            temperature=temperature
        )

        return completion.choices[0].message.content

    def get_categories(self, file_path):
        """
        Reads all the categories from the categories.txt file and returns them as an array.
        :param file_path: File path to the categories.txt file.
        :return: Returns an array containing all the categories from the categories.txt file.
        """
        with open(file_path, 'r') as file:
            return file.read().split("\n")
