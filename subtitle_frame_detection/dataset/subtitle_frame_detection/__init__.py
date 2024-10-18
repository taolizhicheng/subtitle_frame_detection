
__all__ = [
    "SubtitlePairCode"
]


def __dir__():
    return __all__


class SubtitlePairCode:
    NF1_NF2 = 0  # 帧1和帧2都没有字幕
    NF1_F2 = 1   # 帧1没有字幕，帧2有字幕
    F1_NF2 = 2   # 帧1有字幕，帧2没有字幕
    F1_F2_DIFF = 3  # 帧1和帧2都有字幕，但字幕不同
    F1_F2_SAME = 4  # 帧1和帧2都有字幕，且字幕相同