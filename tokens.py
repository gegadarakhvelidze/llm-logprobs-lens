import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Tuple, Any
from rich.text import Text
import json


@dataclass
class RichToken:
    token: str
    logprob: float
    top_logprobs: List["RichToken"] | None = field(
        default=None,
        repr=False,
    )
    bytes: List[int] | None = field(default=None, repr=False)

    @cached_property
    def prob(self):
        return math.exp(self.logprob)

    def __str__(self):
        return f"TokenInfo(token={self.token!r}, prob={self.prob:.3f})"

    def __repr__(self):
        return str(self)

    def to_printable(self) -> str:
        """Format token to show whitespace characters"""
        replacements = {
            " ": "␣",  # Space
            "\n": "↵",  # Newline
            "\t": "⇥",  # Tab
            "\r": "⏎",  # Carriage return
        }
        return "".join(replacements.get(c, c) for c in self.token)

    @classmethod
    def from_logprobs(cls, content_logprobs) -> List["RichToken"]:
        tokens = []

        for token_info in content_logprobs:
            rich_token = RichToken(
                token=token_info.token,
                logprob=token_info.logprob,
                top_logprobs=[
                    RichToken(t.token, t.logprob, bytes=t.bytes)
                    for t in token_info.top_logprobs[1:]
                ],
                bytes=token_info.bytes,
            )
            tokens.append(rich_token)
        return tokens

    def to_rich_colored_text(self):
        # Return default color if logprob is None
        if self.logprob is None:
            return "white"

        # Convert logprob to probability (0 to 1 scale)
        prob = self.prob

        # Calculate red and green components
        # When prob = 0.5, both red and green will be 128
        # When prob = 0, red will be 255 and green will be 0
        # When prob = 1, red will be 0 and green will be 255
        red = int(255 * (1 - prob))
        green = int(255 * prob)

        return f"rgb({red},{green},0)"

    def to_dict(self, ignore_keys: list = []) -> Dict[str, Any]:
        """Convert RichToken to a dictionary for JSON serialization"""
        result = {
            "token": self.token,
            "logprob": self.logprob,
        }
        if self.top_logprobs is not None:
            result["top_logprobs"] = [t.to_dict(ignore_keys=ignore_keys) for t in self.top_logprobs]
        if self.bytes is not None:
            result["bytes"] = self.bytes

        for key in ignore_keys:
            result.pop(key, None)

        return result

    def to_json(self) -> str:
        """Convert RichToken to a JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RichToken":
        """Create a RichToken from a dictionary"""
        top_logprobs = None
        if "top_logprobs" in data:
            top_logprobs = [cls.from_dict(t) for t in data["top_logprobs"]]

        return cls(
            token=data["token"],
            logprob=data["logprob"],
            top_logprobs=top_logprobs,
            bytes=data.get("bytes"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "RichToken":
        """Create a RichToken from a JSON string"""
        return cls.from_dict(json.loads(json_str))


def format_rich_tokens_text(
    tokens: List[RichToken], selected_index: int = None
) -> Text:
    styled_text = Text()
    for i, token_info in enumerate(tokens):
        main_color = token_info.to_rich_colored_text()
        style = f"{main_color}"
        if i == selected_index:
            style += " reverse"  # Highlight selected token
        styled_text.append(token_info.token, style=style)
    return styled_text


def get_average_token_prob(tokens: List[RichToken]) -> float:
    """Get the average prob for a token"""
    return sum(token.prob for token in tokens) / len(tokens)


def get_prob_range(tokens: List[RichToken]) -> Tuple[float, float]:
    """
    Get the minimum and maximum probability in the token sequence.
    Helps identify the most and least confident predictions.
    """
    probs = [token.prob for token in tokens]
    return min(probs), max(probs)


def get_low_prob_tokens_count(tokens: List[RichToken], threshold: float = 0.5) -> int:
    """
    Count tokens with probability below a certain threshold.
    Identifies how many tokens the model was uncertain about.
    """
    return sum(1 for token in tokens if token.prob < threshold)


def get_full_sequence_prob(tokens: List[RichToken]) -> float:
    """
    Calculate the probability of the full token sequence.
    """
    return math.exp(sum(token.logprob for token in tokens))


def get_token_stats(tokens: List[RichToken]) -> Dict[str, float]:
    """
    Get various statistics for a list of tokens.
    """
    token_count = len(tokens)
    avg_prob = get_average_token_prob(tokens)
    min_prob, max_prob = get_prob_range(tokens)
    low_prob_count = get_low_prob_tokens_count(tokens)

    return {
        "token_count": token_count,
        "avg_prob": avg_prob,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "low_prob_count": low_prob_count,
        "low_prob_ratio": low_prob_count / token_count,
    }
