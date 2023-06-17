import os
import unittest

from data.PromptManager import PromptManager
from main import args


class TestPromptManagerPreparation(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)

    def test_prepare_prompt_no_prefix(self):
        prompt = self.manager.prepare_prompt("apple", length=20, detail="long", complexity="very complex", prefix="")
        self.assertEqual('Give me 20 very detailed and unique descriptions of an apple. Do not include the word '
                         '"apple" or any of its variations in your response. Use very complex language in your '
                         'response.', prompt)

    def test_prepare_prompt_prefix_it(self):
        prompt = self.manager.prepare_prompt("apple", length=20, detail="long", complexity="very complex", prefix="it")
        self.assertEqual(
            'Give me 20 very detailed and unique descriptions of an apple. Do not include the word "apple" '
            'or any of its variations in your response. Use very complex language in your response. '
            'Start all your responses with "It".', prompt)

    def test_prepare_prompt_detail_very_short(self):
        prompt = self.manager.prepare_prompt("ear", length=20, detail="very short", complexity="very complex", prefix="it")
        print(prompt)

    def test_prepare_prompt_detail_short(self):
        prompt = self.manager.prepare_prompt("ear", length=20, detail="short", complexity="very complex", prefix="it")
        print(prompt)

    def test_prepare_prompt_detail(self):
        prompt = self.manager.prepare_prompt("ear", length=20, detail=None, complexity="very complex", prefix="it")
        print(prompt)

    def test_prepare_prompt_detail_long(self):
        prompt = self.manager.prepare_prompt("ear", length=20, detail="long", complexity="very complex", prefix="it")
        print(prompt)

    def test_prepare_prompt_detail_very_long(self):
        prompt = self.manager.prepare_prompt("ear", length=20, detail="very long", complexity="very complex", prefix="it")
        print(prompt)


class TestPromptManagerPrompting(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)

    def test_prepare_prompt_temperature_1(self):
        prompt = self.manager.prepare_prompt("yacht", length=1, detail="long", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt, temperature=0.2)
        print(response)

    def test_prepare_prompt_temperature_2(self):
        prompt = self.manager.prepare_prompt("yacht", length=1, detail="long", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt, temperature=0.6)
        print(response)

    def test_prepare_prompt_temperature_3(self):
        prompt = self.manager.prepare_prompt("yacht", length=1, detail="long", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt, temperature=1.0)
        print(response)

    def test_prepare_prompt_detail_very_short(self):
        prompt = self.manager.prepare_prompt("ear", length=1, detail="very short", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt)
        print(response)

    def test_prepare_prompt_detail_short(self):
        prompt = self.manager.prepare_prompt("ear", length=1, detail="short", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt)
        print(response)

    def test_prepare_prompt_detail(self):
        prompt = self.manager.prepare_prompt("ear", length=1, detail=None, complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt)
        print(response)

    def test_prepare_prompt_detail_long(self):
        prompt = self.manager.prepare_prompt("ear", length=1, detail="long", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt)
        print(response)

    def test_prepare_prompt_detail_very_long(self):
        prompt = self.manager.prepare_prompt("ear", length=1, detail="very long", complexity="complex", prefix="")
        response = self.manager.make_safe_prompt(prompt)
        print(response)
