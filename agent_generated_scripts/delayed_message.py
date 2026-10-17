"""Print a message after a configurable delay."""

import argparse
import math
import time


def delayed_print(message: str, *, n: float = 1.0) -> None:
    """Wait n seconds, then print message."""
    if not math.isfinite(n) or n < 0:
        raise ValueError("n must be a finite, nonnegative number of seconds")
    time.sleep(n)
    print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", help="Message to print after the delay")
    parser.add_argument("--n", type=float, default=1.0, help="Seconds to wait (default: 1)")
    args = parser.parse_args()
    if not math.isfinite(args.n) or args.n < 0:
        parser.error("--n must be a finite, nonnegative number of seconds")
    delayed_print(args.message, n=args.n)
