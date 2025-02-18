import math
import os
import readline
import sys
import termios
import tty
from getpass import getpass
from typing import Dict, List

from dotenv import load_dotenv
from openai import Client
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

load_dotenv()
model_name = os.getenv("OPENAI_MODEL_NAME")
console = Console()

# Configure readline
HISTORY_FILE = os.path.expanduser("~/.uncertainty_history")
try:
    readline.read_history_file(HISTORY_FILE)
except FileNotFoundError:
    pass

readline.set_history_length(1000)  # Keep last 1000 commands
readline.parse_and_bind("set editing-mode emacs")  # Enable arrow keys


def get_prompt_with_history(prompt: str) -> str:
    try:
        user_input = input(prompt)
        readline.write_history_file(HISTORY_FILE)
        return user_input
    except (EOFError, KeyboardInterrupt):  # Handle Ctrl+D and Ctrl+C
        print()
        return "exit"


class TokenInfo:
    def __init__(self, token: str, logprob: float, alternatives: List[Dict]):
        self.token = token
        self.logprob = logprob
        self.alternatives = alternatives


# Global storage for token information
token_registry = {}
current_token_id = 0


def fetch_llm_response(prompt: str):
    client = Client(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    return client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        frequency_penalty=0.0,
        logprobs=True,
        top_logprobs=10,
        stream=False,
    )


def map_logprob_to_color(logprob):
    # Return default color if logprob is None
    if logprob is None:
        return "white"

    # Convert logprob to probability (0 to 1 scale)
    prob = math.exp(logprob)

    # Calculate red and green components
    # When prob = 0.5, both red and green will be 128
    # When prob = 0, red will be 255 and green will be 0
    # When prob = 1, red will be 0 and green will be 255
    red = int(255 * (1 - prob))
    green = int(255 * prob)

    return f"rgb({red},{green},0)"


def format_token(token: str) -> str:
    """Format token to show whitespace characters"""
    replacements = {
        " ": "␣",  # Space
        "\n": "↵",  # Newline
        "\t": "⇥",  # Tab
        "\r": "⏎",  # Carriage return
    }
    return "".join(replacements.get(c, c) for c in token)


def show_alternatives(token_info: TokenInfo):
    """Display alternative tokens in a pretty table"""
    table = Table(title=f"Alternatives for '{format_token(token_info.token)}'")
    table.add_column("Token", style="cyan")
    table.add_column("Visual", style="blue")
    table.add_column("Probability", style="magenta")

    # Add the main token
    main_prob = math.exp(token_info.logprob)
    table.add_row(
        repr(token_info.token),
        format_token(token_info.token),
        f"{main_prob:.4f} (selected)",
    )

    # Add alternative tokens
    for alt in token_info.alternatives:
        prob = math.exp(alt.logprob)
        table.add_row(repr(alt.token), format_token(alt.token), f"{prob:.4f}")

    console.print(table)


def get_key():
    """Get a single keypress from terminal"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def display_response_with_color(response):
    global current_token_id
    token_registry.clear()
    current_token_id = 0
    selected_index = 0

    # Store tokens for navigation
    tokens = []
    content_logprobs = response.choices[0].logprobs.content

    for token_info in content_logprobs:
        token_id = f"token_{current_token_id}"
        token_registry[token_id] = TokenInfo(
            token=token_info.token,
            logprob=token_info.logprob,
            alternatives=token_info.top_logprobs[1:],
        )
        tokens.append((token_info, token_id))
        current_token_id += 1

    while True:
        # Display tokens with current selection highlighted
        styled_text = Text()
        for i, (token_info, token_id) in enumerate(tokens):
            main_color = map_logprob_to_color(token_info.logprob)
            style = f"{main_color}"
            if i == selected_index:
                style += " reverse"  # Highlight selected token
            styled_text.append(token_info.token, style=style)

        # Clear screen and show current state
        console.clear()
        console.print(styled_text)
        console.print(
            "\n[dim]← → to navigate | ↵ to show alternatives | q to quit[/dim]"
        )

        # Handle keyboard input
        key = get_key()
        if key == "\x1b":  # Arrow keys start with escape
            next_two = sys.stdin.read(2)
            if next_two == "[C":  # Right arrow
                selected_index = min(selected_index + 1, len(tokens) - 1)
            elif next_two == "[D":  # Left arrow
                selected_index = max(selected_index - 1, 0)
        elif key in ("q", "Q", "ქ"):  # Quit
            break
        elif key == "\r":  # Enter
            token_id = tokens[selected_index][1]
            console.print("\n")
            show_alternatives(token_registry[token_id])
            getpass("\nPress Enter to continue...")


def main():
    while True:
        console.clear()
        prompt_text = get_prompt_with_history("Enter prompt (or 'exit' to quit): ")
        if prompt_text.lower() in ("exit", "q", "quit"):
            break
        response = fetch_llm_response(prompt_text)
        display_response_with_color(response)


if __name__ == "__main__":
    main()
