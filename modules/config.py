# -*- coding: utf-8 -*-
import openai

from colorama import just_fix_windows_console, Fore, Style
just_fix_windows_console()
from dotenv import dotenv_values
from modules import key


def success():
    key.success()

    envs = dotenv_values(".env")
    if "OPENAI_API_HTTP_PROXY" not in envs:
        print(f"{Fore.RED}-> No env 'OPENAI_API_HTTP_PROXY' found in .env file\n{Style.RESET_ALL}")
        exit(-1)
    else:
        openai.proxy = envs["OPENAI_API_HTTP_PROXY"]
        print(f"{Fore.YELLOW}-> Loaded openai api http proxy:{openai.proxy}\n{Style.RESET_ALL}")
    openai.log = "info"

    print(f"{Fore.GREEN}-> Loaded default settings\n{Style.RESET_ALL}")
