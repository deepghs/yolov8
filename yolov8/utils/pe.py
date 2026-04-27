"""Pretty-print helper for compact human-readable numbers."""


def float_pe(v: float, gnum: int = 3) -> str:
    """Render ``v`` as the shortest of ``{plain, k, M, G}`` formats.

    Each candidate is formatted with ``{:.<gnum>g}`` against a
    progressively bigger denominator (``1``, ``1e3``, ``1e6``,
    ``1e9``) and the candidate with the fewest characters wins. Used
    by :mod:`yolov8.list` to fit FLOPs / params into a compact README
    table.

    :param v: Numeric value to format.
    :type v: float
    :param gnum: Significant digits for the underlying ``%g`` format.
        Defaults to ``3``.
    :type gnum: int
    :returns: The shortest candidate string, e.g. ``"42"`` /
        ``"3.14k"`` / ``"2.5M"`` / ``"1.2G"``.
    :rtype: str

    Example::

        >>> from yolov8.utils import float_pe
        >>> float_pe(3_000_000)
        '3M'
        >>> float_pe(0.025)
        '0.025'
    """
    texts = [
        f'{v:.{gnum}g}',
        f'{v / 1e3:.{gnum}g}k',
        f'{v / 1e6:.{gnum}g}M',
        f'{v / 1e9:.{gnum}g}G',
    ]

    best_text = None
    for text in texts:
        if best_text is None or len(text) < len(best_text):
            best_text = text

    return best_text
