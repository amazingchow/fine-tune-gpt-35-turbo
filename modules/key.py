# -*- coding: utf-8 -*-
import openai

from colorama import Fore, Style
from dotenv import dotenv_values


def success():
    envs = dotenv_values(".env")
    if "OPENAI_API_KEY" not in envs:
        print(f"{Fore.RED}-> No env 'OPENAI_API_KEY' found in .env file\n{Style.RESET_ALL}")
        exit(-1)
    else:
        openai.api_key = envs["OPENAI_API_KEY"]
        print(f"{Fore.YELLOW}-> Loaded openai api key:{openai.api_key}\n{Style.RESET_ALL}")
