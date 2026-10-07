import sys

from .cli import main

# The guard matters: the worker process starts with multiprocessing's spawn
# method, which imports the main module again in the child. Without it, the
# child would run the whole command line again.
if __name__ == "__main__":
    sys.exit(main())
