import math
import os
import readline
import sys
import termios
import tty
from getpass import getpass

from dotenv import load_dotenv
from openai import Client
from rich.console import Console
from rich.table import Table

from tokens import RichToken, format_rich_tokens_text

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


def fetch_llm_response(prompt: str, n=1):
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
        n=n,
    )


def show_alternatives(token_info: RichToken):
    """Display alternative tokens in a pretty table"""
    table = Table(title=f"Alternatives for '{token_info.to_printable()}'")
    table.add_column("Token", style="cyan")
    table.add_column("Visual", style="blue")
    table.add_column("Probability", style="magenta")

    # Add the main token
    main_prob = math.exp(token_info.logprob)
    table.add_row(
        repr(token_info.token),
        token_info.to_printable(),
        f"{main_prob:.4f} (selected)",
    )

    # Add alternative tokens
    for alt in token_info.top_logprobs:
        prob = math.exp(alt.logprob)
        table.add_row(repr(alt.token), alt.to_printable(), f"{prob:.4f}")

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
    selected_index = 0

    tokens = RichToken.from_logprobs(response.choices[0].logprobs.content)

    while True:
        styled_text = format_rich_tokens_text(tokens, selected_index)

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
            console.print("\n")
            show_alternatives(tokens[selected_index])
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
