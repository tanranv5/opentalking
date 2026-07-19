"""_StreamingTextCollector 流式合并回归测试。

复现并锁死修复：DashScope 流式回调 ``get_sentence()`` 返回 dict（单句），
多句长语音必须累积全部已定稿分句 + 尾部 partial，不得只剩最后一句。
"""

from __future__ import annotations

import queue
from typing import Any

from opentalking.providers.stt.dashscope.adapter import _StreamingTextCollector


class _FakeResult:
    """最小化 RecognitionResult 桩：仅提供 get_sentence()。"""

    def __init__(self, sentence):
        self._sentence = sentence

    def get_sentence(self):
        return self._sentence


def _partial(text: str) -> Any:
    return _FakeResult({"text": text, "begin_time": 0, "end_time": None})


def _sentence_end(text: str, begin: int, end: int) -> Any:
    return _FakeResult({"text": text, "begin_time": begin, "end_time": end})


def _list_result(sentences: list[dict]) -> Any:
    return _FakeResult(sentences)


def test_multi_sentence_stream_keeps_all_segments():
    """三句长语音：合并结果必须包含全部三句（原 bug 只剩最后一句）。"""
    c = _StreamingTextCollector()
    c.on_event(_partial("今天"))
    c.on_event(_partial("今天上午"))
    c.on_event(_sentence_end("今天上午我去了公司。", 0, 3000))
    c.on_event(_partial("下午"))
    c.on_event(_sentence_end("下午开了三个会。", 3000, 6000))
    c.on_event(_partial("晚上"))
    c.on_event(_sentence_end("晚上才回家。", 6000, 9000))
    assert c.combined_text() == "今天上午我去了公司。下午开了三个会。晚上才回家。"


def test_trailing_partial_not_dropped():
    """最后一句未到 sentence_end 就 stop：尾部 partial 必须保留。"""
    c = _StreamingTextCollector()
    c.on_event(_sentence_end("第一句。", 0, 2000))
    c.on_event(_partial("第二句还没说"))
    assert c.combined_text() == "第一句。第二句还没说"


def test_duplicate_sentence_end_not_double_appended():
    """SDK 重发同一句尾（相同 begin/end_time）不得重复拼接。"""
    c = _StreamingTextCollector()
    c.on_event(_sentence_end("重复句。", 0, 1500))
    c.on_event(_sentence_end("重复句。", 0, 1500))
    assert c.combined_text() == "重复句。"


def test_user_really_repeats_same_text_kept():
    """用户真的连说两遍相同内容（时间区间不同）：两遍都要保留。"""
    c = _StreamingTextCollector()
    c.on_event(_sentence_end("我不知道。", 0, 1500))
    c.on_event(_sentence_end("我不知道。", 1600, 3000))
    assert c.combined_text() == "我不知道。我不知道。"


def test_sync_list_path_still_works():
    """同步 call() 路径返回 list 的兼容性。"""
    c = _StreamingTextCollector()
    c.on_event(
        _list_result([
            {"text": "第一句。", "begin_time": 0, "end_time": 1000},
            {"text": "第二句。", "begin_time": 1000, "end_time": 2000},
        ])
    )
    assert c.combined_text() == "第一句。第二句。"


def test_event_protocol_no_final_from_collector():
    """collector 只发 partial / segment_final；整段唯一终态 transcript.final 由 API 层发。"""
    q: queue.Queue = queue.Queue()
    c = _StreamingTextCollector(q)
    c.on_event(_partial("你好"))
    c.on_event(_sentence_end("你好吗？", 0, 1200))
    c.on_event(_partial("我很好"))

    events = []
    while not q.empty():
        events.append(q.get())

    types = [e["type"] for e in events]
    assert "transcript.final" not in types
    assert types == ["transcript.partial", "transcript.segment_final", "transcript.partial"]
    # partial / segment_final 均携带累计全文预览
    assert events[1]["text"] == "你好吗？"
    assert events[1]["is_final"] is False
    assert events[2]["text"] == "你好吗？我很好"
