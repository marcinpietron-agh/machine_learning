import torch

from quality_prompts.exemplars import ExemplarStore, Exemplar
from quality_prompts.utils.llm import get_embedding

import os

from quality_prompts.prompt import QualityPrompt
from quality_prompts.utils.llm import llm_call

#environment API KEY

directive = """Solve the given math problem"""
import json

#prompt = QualityPrompt(directive=directive, additional_information=additional_information, exemplar_store=exemplar_store)
#prompt = QualityPrompt(directive=directive, exemplar_store=exemplar_store)
prompt = QualityPrompt(directive=directive)

input_text = """Jackson is planting tulips. He can fit 6 red tulips in a row and 8 blue tulips in a row. If Jackson buys 36 red tulips and 24 blue tulips, how many rows of flowers will he plant?"""

prompt.tabular_chain_of_thought_prompting(input_text=input_text)
compiled_quality_prompt = prompt.compile()
print(compiled_quality_prompt)

messages = [{"role" : "system", "content" : compiled_quality_prompt},
            {"role" : "user", "content" : input_text}]
response = llm_call(messages)
print(response)


directive = """Solve the given math problem"""
prompt = QualityPrompt(directive=directive, additional_information="",)

#input_text = """What happens to the pressure, P, of an ideal gas if the temperature is increased by a factor of 2 and the  volume is increased by a factor of 8 ?"""

input_text = """Solve the given math problem
        Question: How does the pressure of an ideal gas change when the temperature and volume are altered according to specific factors?
                                            Answer: According to the ideal gas law, the pressure of an ideal gas is directly proportional to its temperature and inversely proportional to its volume. This relationship can be expressed by the formula:

\[ PV = nRT \]

where:
- \( P \) is the pressure of the gas
- \( V \) is the volume of the gas
- \( n \) is the number of moles of gas
- \( R \) is the ideal gas constant
- \( T \) is the temperature of the gas in kelvin

If we alter the temperature and volume of the gas according to specific factors, the pressure will change as follows:

1. If the volume of the gas is decreased while keeping the temperature constant, the pressure will increase. This is known as Boyle's Law, which states that pressure and volume are inversely proportional when temperature is constant.

2. If the temperature of the gas is increased while keeping the volume constant, the pressure will also increase. This is known as Charles's Law, which states that pressure and temperature are directly proportional when volume is constant.

3. If both the temperature and volume of the gas are changed, the overall effect on pressure will depend on the specific changes made. However, in general, increasing temperature and decreasing volume will lead to a greater increase in pressure.

In summary, the pressure of an ideal gas will change in response to alterations in temperature and volume according to the relationships described by Boyle's Law and Charles's Law.
"""



prompt.step_back_prompting(input_text=input_text)
compiled_quality_prompt = prompt.compile()
print(compiled_quality_prompt)

messages = [{"role" : "system", "content" : compiled_quality_prompt},
            {"role" : "user", "content" : input_text}]
response = llm_call(messages)
print(response)