import json
import os
import argparse


def update_version(mode: str, version: int = None, filename='version.json') -> str:
    """
    Version manager:
      - mode = "train": create a new major version (vX.0)
      - mode = "sample": increment the minor version for a given major (vX.Y)

    Args:
        mode (str): either "train" or "sample"
        version (int, optional): major version number (required only in sample mode)
        filename (str): JSON file path to store version info

    Returns:
        str: version string in format "vX.Y"
    """
    # Load or initialize version data
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # ---- Train mode: create new major version ----
    if mode == "train":
        new_major = (max(map(int, data.keys())) + 1) if data else 0
        data[str(new_major)] = 0
        version_str = f"v{new_major}.0"

    # ---- Sample mode: increment existing minor version ----
    elif mode == "sample":
        if version is None:
            raise ValueError("In 'sample' mode, you must provide --version <major_version>.")
        key = str(version)
        data[key] = data.get(key, 0) + 1
        version_str = f"v{key}.{data[key]}"

    else:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'train' or 'sample'.")

    # Save updated version info
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    return version_str


def main_entry():
    """
    CLI entry point for managing training/sampling runs.
    Example usage:
        python run.py --mode train
        python run.py --mode sample --version 3
    """
    parser = argparse.ArgumentParser(description="Version-controlled training and sampling launcher.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "sample"],
                        help="Select run mode: 'train' to create a new version, 'sample' to use an existing one.")
    parser.add_argument("--version", type=int, default=None,
                        help="Major version number (required only for 'sample' mode).")
    args = parser.parse_args()

    # Determine main entry point
    if args.mode == "train":
        from train import main
    elif args.mode == "sample":
        from sample import main
    else:
        raise ValueError("Invalid mode specified.")

    version_str = update_version(args.mode, args.version)
    print(f"Running mode={args.mode}, version={version_str}")
    main(version_str)


if __name__ == "__main__":
    main_entry()
