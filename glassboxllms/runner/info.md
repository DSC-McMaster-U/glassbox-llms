Welcome to the runner.

It orchestrates and is responsible for calling experiments.

`tracking.py` and `config.py` are just backend stuff to set up tracking and data structures respectively. `cli.py` is the outermost input layer. It is what handles the command line input and calls `core.py`'s run and setup functions. `core.py` stores the main logic.
