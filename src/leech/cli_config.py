"""
Rich-click styling configuration and branding for the leech CLI.
"""

import rich_click as click
from rich.console import Console
from rich.panel import Panel

# ASCII Logo
LOGO = """
   ___
  (O O)    LEECH
  <VVV>    Learning Enhanced Electrical
   |||     Classifiers from Hanopore signals
   |||
    V
"""

console = Console()


def display_logo():
    """Display the ASCII logo in a panel."""
    console.print(Panel(LOGO, border_style="cyan", padding=(0, 2)))


def configure_rich_click():
    """Apply rich-click styling for beautiful help display."""
    click.rich_click.USE_RICH_MARKUP = True
    click.rich_click.SHOW_ARGUMENTS = True
    click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
    click.rich_click.MAX_WIDTH = 100

    # Error styling
    click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
    click.rich_click.ERRORS_SUGGESTION = "Try running the '--help' flag for more information."
    click.rich_click.ERRORS_EPILOGUE = ""

    # Color styling for help elements
    click.rich_click.STYLE_OPTION = "bold cyan"
    click.rich_click.STYLE_ARGUMENT = "bold yellow"
    click.rich_click.STYLE_COMMAND = "bold green"
    click.rich_click.STYLE_SWITCH = "bold blue"
    click.rich_click.STYLE_METAVAR = "bold yellow"
    click.rich_click.STYLE_METAVAR_SEPARATOR = "dim"
    click.rich_click.STYLE_HEADER_TEXT = "bold magenta"
    click.rich_click.STYLE_EPILOG_TEXT = "dim"
    click.rich_click.STYLE_FOOTER_TEXT = "dim"
    click.rich_click.STYLE_USAGE = "bold yellow"
    click.rich_click.STYLE_USAGE_COMMAND = "bold"
    click.rich_click.STYLE_DEPRECATED = "red"
    click.rich_click.STYLE_HELPTEXT_FIRST_LINE = ""
    click.rich_click.STYLE_HELPTEXT = "dim"
    click.rich_click.STYLE_OPTION_HELP = ""
    click.rich_click.STYLE_OPTION_DEFAULT = "dim italic"
    click.rich_click.STYLE_REQUIRED_SHORT = "bold red"
    click.rich_click.STYLE_REQUIRED_LONG = "bold red"

    # Table and panel styling
    click.rich_click.STYLE_OPTIONS_TABLE_LEADING = 1
    click.rich_click.STYLE_OPTIONS_TABLE_PAD_EDGE = True
    click.rich_click.STYLE_OPTIONS_TABLE_PADDING = (0, 1)
    click.rich_click.STYLE_COMMANDS_TABLE_LEADING = 1
    click.rich_click.STYLE_COMMANDS_TABLE_PAD_EDGE = True
    click.rich_click.STYLE_COMMANDS_TABLE_PADDING = (0, 1)

    # Additional features
    click.rich_click.SHOW_METAVARS_COLUMN = True
    click.rich_click.APPEND_METAVARS_HELP = True
    click.rich_click.USE_CLICK_SHORT_HELP = True
