"""
Compatibility wrapper for older entry points.
Redirects to the new single-page MCA dashboard in dashboard/main.py
"""

from dashboard.main import main as _main


def main():  # pragma: no cover - thin wrapper
    _main()


if __name__ == "__main__":
    main()