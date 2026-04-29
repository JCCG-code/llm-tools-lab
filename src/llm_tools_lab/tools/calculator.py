def add(a: float, b: float) -> float:
    """
    Add two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The sum of the two numbers
    """
    return a + b


def subtract(a: float, b: float) -> float:
    """
    Subtract two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The subtraction of the two numbers
    """
    return a - b


def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The product of the two numbers
    """
    return a * b


def divide(a: float, b: float) -> float:
    """
    Divide two numbers.

    Args:
        a: The first number
        b: The second number

    Returns:
        The division of the two numbers

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def power(base: float, exp: float) -> float:
    """
    Calculate the power of a number

    Args:
        base: The base number
        exp: The exponent

    Returns:
        Base raised to the power of exponent
    """
    return base**exp


MATH_TOOLS = [add, subtract, multiply, divide, power]
