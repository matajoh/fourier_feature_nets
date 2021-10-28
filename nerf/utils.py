from progress.bar import Bar


class ETABar(Bar):
    suffix = '%(percent).1f%% - %(eta)ds'
    bar_prefix = ' '
    bar_suffix = ' '
    empty_fill = '∙'
    fill = '█'
