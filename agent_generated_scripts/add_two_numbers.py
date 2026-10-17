#!/usr/bin/env python3
import sys

def add_two_numbers(a, b):
    """Add two numbers and return the sum."""
    return a + b

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_two_numbers.py <number1> <number2>")
        sys.exit(1)
    
    try:
        num1 = float(sys.argv[1])
        num2 = float(sys.argv[2])
        result = add_two_numbers(num1, num2)
        print(f"Sum of {num1} and {num2} is: {result}")
    except ValueError:
        print("Error: Please provide two valid numbers as arguments.")
        sys.exit(1)
