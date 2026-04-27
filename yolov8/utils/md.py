"""Tiny markdown-table parser used by the README-aggregation script."""
import pandas as pd


def markdown_to_df(markdown_text: str) -> pd.DataFrame:
    """Parse a single markdown table into a :class:`pandas.DataFrame`.

    Strict format: header row, ``|---|---|`` separator, then data
    rows. No support for nested pipes inside cells or multi-line
    cells - exactly the format :mod:`yolov8.list` rewrites in HF
    README files.

    :param markdown_text: Multi-line string containing a single
        markdown table; the surrounding whitespace is stripped before
        parsing.
    :type markdown_text: str
    :returns: A DataFrame whose columns come from the header row and
        whose rows come from the body cells, all stored as strings.
    :rtype: pandas.DataFrame

    Example::

        >>> from yolov8.utils import markdown_to_df
        >>> tbl = "| a | b |\\n| --- | --- |\\n| 1 | 2 |"
        >>> markdown_to_df(tbl).iloc[0].to_dict()
        {'a': '1', 'b': '2'}
    """
    lines = markdown_text.strip().split('\n')
    header = lines[0]
    data = lines[2:]
    header = header.strip('|').split('|')
    header = [col.strip() for col in header]
    data = [row.strip('|').split('|') for row in data]
    data = [[cell.strip() for cell in row] for row in data]

    df = pd.DataFrame(
        columns=header,
        data=data
    )
    return df
