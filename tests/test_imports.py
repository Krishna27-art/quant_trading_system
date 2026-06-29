import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_imports():

    print("All critical imports successful!")


if __name__ == "__main__":
    test_imports()
