#!/usr/bin/env python3
"""Sum calculator that takes two arguments and returns their sum."""

def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python sum_calculator.py <num1> <num2>")
        sys.exit(1)
    
    try:
        num1 = float(sys.argv[1])
        num2 = float(sys.argv[2])
        result = calculate_sum(num1, num2)
        print(f"Sum of {num1} and {num2} is: {result}")
    except ValueError:
        print("Error: Please provide two valid numbers as arguments.")
        sys.exit(1)