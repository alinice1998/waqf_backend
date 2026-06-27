"""Allow running waqf_backend as a module: python -m waqf_backend."""

from waqf_backend.cli import cli

if __name__ == "__main__":
    cli()
