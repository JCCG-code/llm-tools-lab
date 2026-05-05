from collections.abc import Callable

from llm_tools_lab.tools.calculator import MATH_TOOLS
from llm_tools_lab.tools.strings import STRING_TOOLS

# Keywords mapping
MATH_KEYWORDS = [
    "calculate",
    "calculation",
    "what is",
    "compute",
    "sum",
    "plus",
    "add",
    "multiply",
    "times",
    "divide",
    "subtract",
    "minus",
    "division",
    "subtraction",
    "power",
    "result of",
]
STRING_KEYWORDS = ["uppercase", "reverse", "words", "capitalize"]


def get_tools_for_message(message: str) -> list[Callable]:
    # Sanitize message
    clean = message.lower().replace("?", "").replace("!", "").replace(",", "")
    words = clean.split()
    # Detects keywords
    has_math = any(word in MATH_KEYWORDS for word in words)
    has_string = any(word in STRING_KEYWORDS for word in words)
    # Returns
    if has_math and has_string:
        return MATH_TOOLS + STRING_TOOLS
    elif has_math:
        return MATH_TOOLS
    elif has_string:
        return STRING_TOOLS
    else:
        return []
