from rich.console import Console

_console = Console()

def success(message: str):
    """Print a success message in green."""
    _console.print(message, style="bold green")

def error(message: str):
    """Print an error message in red."""
    _console.print(message, style="bold red")

def info(message: str):
    """Print a neutral info message (default styling)."""
    _console.print(message)
