import asyncio
import sys

from pipeline import run

def main():
    try:
        asyncio.run(run())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
