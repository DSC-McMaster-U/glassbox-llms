import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .core import Runner

# load config -> initialize runner -> set up runner -> run runner


def main():
    parser = argparse.ArgumentParser(description="Glassbox Runner CLI")
    # TODO: Think about adding more arguments/shorthand (-)?
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (YAML/JSON)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config without running the experiment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        cfg = load_config(args.config)
        logging.info(f"Loaded configuration from {args.config}")

        if args.dry_run:
            logging.info("Dry run enabled. Configuration validated successfully.")
            print(cfg.model_dump_json(indent=2))
            return

        runner = Runner(cfg)
        runner.setup()
        runner.run()
        logging.info("Experiment completed successfully.")

    except Exception as e:
        logging.error(f"Error running experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
