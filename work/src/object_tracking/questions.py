"""Recognize narrow file-history questions without treating them as writes."""

import re


_RECENT_CREATION_QUESTIONS = (
    r"(?:what|which) (?:file|filename) (?:(?:did|have|has) )?"
    r"(?:i|we|you) (?:just |recently )?(?:create|created)",
    r"(?:what|which) file (?:was|has been) (?:just |recently )?created",
    r"what (?:is|was) (?:the )?(?:name|filename|path) of (?:the )?file "
    r"(?:(?:i|we|you) (?:just |recently )?created|(?:just |recently )?created)",
    r"what (?:is|was) (?:the )?(?:(?:last|latest|most recently) (?:created )?file|"
    r"(?:last|latest) file (?:i|we|you) created|file (?:i|we|you) (?:just |recently )?created)",
    r"(?:我|你|我们)?(?:刚才|刚刚|刚|最近)(?:创建|新建)(?:了|的)?(?:什么|哪个|哪一个)(?:文件|文档)",
    r"(?:我|你|我们)?(?:刚才|刚刚|刚|最近)(?:创建|新建)(?:的)?(?:文件|文档)"
    r"(?:是|叫|叫什么|的名字是|的名称是)(?:什么|哪个|哪一个|什么名字|什么名称)?",
)


_PATH_QUESTION = (r"(?:(?:what (?:is|was)|show(?: me)?|tell me) )?"
                  r"(?:the )?(?:name|filename|path) (?:of|for) ")
_FIRST_CREATED_FILE = (
    r"(?:(?:the )?(?:first|earliest) (?:created )?file"
    r"(?: (?:that )?(?:i|we|you) created)?|"
    r"(?:the )?file (?:that )?(?:i|we|you) (?:created first|first created|initially created))"
)
_FIRST_CREATION_QUESTIONS = (
    _PATH_QUESTION + _FIRST_CREATED_FILE,
    r"(?:what|which) (?:file|filename) (?:(?:did|have|has) )?(?:i|we|you) "
    r"(?:create(?:d)? first|first create(?:d)?|initially create(?:d)?)",
    r"(?:what|which) (?:is|was) " + _FIRST_CREATED_FILE,
    r"(?:what|which) file (?:was|has been) (?:first|initially) created",
    r"(?:我|你|我们)?(?:最先|最早|首先|第一个)(?:创建|新建)(?:了|的)?"
    r"(?:什么|哪个|哪一个)(?:文件|文档)",
    r"(?:我|你|我们)?(?:最先|最早|首先|第一个)(?:创建|新建)(?:的)?(?:文件|文档)"
    r"(?:是|叫|叫什么|的名字是|的名称是|的路径是|的路径是什么|的路径在哪里)"
    r"(?:什么|哪个|哪一个|什么名字|什么名称)?",
    r"(?:显示|查看|告诉我)(?:我|你|我们)?(?:最先|最早|首先|第一个)(?:创建|新建)"
    r"(?:的)?(?:文件|文档)(?:的)?(?:路径|名字|名称)",
)
_FIRST_USE_QUESTIONS = (
    _PATH_QUESTION + r"(?:the )?file (?:that )?(?:i|we|you) (?:used first|first used)",
    r"(?:what|which) file (?:(?:did|have|has) )?(?:i|we|you) (?:use(?:d)? first|first use(?:d)?)",
    r"(?:我|你|我们)?(?:最先|最早|第一个)使用(?:的)?(?:文件|文档)"
    r"(?:是哪个|是什么|的路径是什么|的路径在哪里)",
    r"(?:显示|查看|告诉我)(?:我|你|我们)?(?:最先|最早|第一个)使用(?:的)?"
    r"(?:文件|文档)(?:的)?路径",
)
_FIRST_CREATION_ANSWERS = (
    r"(?:the )?(?:one|file) (?:that )?(?:i|we|you) (?:created first|first created)",
    r"(?:the )?(?:first|earliest) (?:one|file)(?: (?:that )?(?:i|we|you) created)?",
    _FIRST_CREATED_FILE,
    r"(?:我|你|我们)?(?:最先|最早|第一个)(?:创建|新建)(?:的)?(?:那个|那一个)?(?:文件|文档)?",
)


def _normalize(request):
    text = " ".join(request.strip().lower().split()).rstrip(" ?？.!！。")
    while True:
        stripped = re.sub(r"^(?:please |help me |(?:can|could) you |tell me |remind me |请)", "", text)
        if stripped == text:
            return text
        text = stripped


def creation_question_order(request):
    """Select a creation-history query only for a complete read-only question."""
    text = _normalize(request)
    if any(re.fullmatch(pattern, text) for pattern in _FIRST_CREATION_QUESTIONS):
        return "first"
    recent_path = (_PATH_QUESTION + r"(?:the )?file (?:that )?(?:i|we|you) "
                   r"(?:just |recently )?created")
    if any(re.fullmatch(pattern, text) for pattern in (*_RECENT_CREATION_QUESTIONS, recent_path)):
        return "latest"
    return None


def is_first_use_question(request):
    """'Used first' needs a meaning clarification, not a recent-focus lookup."""
    text = _normalize(request)
    return any(re.fullmatch(pattern, text) for pattern in _FIRST_USE_QUESTIONS)


def creation_answer_order(answer):
    """Resolve a short reply only inside an existing file-history question."""
    text = _normalize(answer)
    if any(re.fullmatch(pattern, text) for pattern in _FIRST_CREATION_ANSWERS):
        return "first"
    return creation_question_order(answer)


def is_file_history_question(request):
    return creation_question_order(request) is not None or is_first_use_question(request)


def is_recent_creation_question(request):
    """Match a complete question so a chained action keeps the command workflow."""
    return creation_question_order(request) == "latest"
