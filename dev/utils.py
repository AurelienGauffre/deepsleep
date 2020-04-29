from pathlib import Path

import yaml
from colorama import Fore, Style


def print_blue_bold(message: str) -> str:
    print(f'{Fore.BLUE}{Style.BRIGHT}{message}{Style.RESET_ALL}')


def print_warning(message: str) -> str:
    print(f'{Fore.YELLOW}{message}{Style.RESET_ALL}')
