"""Utility module."""

from progress.bar import Bar


class ETABar(Bar):
    """Progress bar that displays the estimated time of completion."""
    suffix = "%(percent).1f%% - %(eta)ds"
    bar_prefix = " "
    bar_suffix = " "
    empty_fill = "∙"
    fill = "█"

    def info(self, text: str):
        self.suffix = "%(percent).1f%% - %(eta)ds {}".format(text)
