def uppercase(text: str) -> str:
    """
    Converts the text to uppercase

    Args:
        text: The required text

    Returns:
        The same text in uppercase
    """
    return text.upper()


def reverse(text: str) -> str:
    """
    Reverse a string character by character

    Args:
        text: The required text

    Returns:
        The same flipped text
    """
    return text[::-1]


def count_words(text: str) -> int:
    """
    Count the number of words in a text

    Args:
        text: The required text

    Returns:
        A number of words of the text
    """
    return len(text.split())


def capitalize(text: str) -> str:
    """
    Capitalize the text

    Args:
        text: The required text

    Returns:
        The same text capitalized
    """
    return text.capitalize()


STRING_TOOLS = [uppercase, reverse, count_words, capitalize]
