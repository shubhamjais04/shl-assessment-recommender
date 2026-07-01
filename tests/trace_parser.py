"""
Parses the GenAI_SampleConversations/*.md trace files into:
  - a list of user messages in order (to replay against /chat)
  - the expected final shortlist (names) from the last recommendation table
    in the file, used as ground truth for Recall@10.
"""

import re
from pathlib import Path

TRACES_DIR = Path(__file__).parent.parent / "traces" / "GenAI_SampleConversations"

USER_BLOCK_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)
TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|\s*(.+?)\s*\|", re.MULTILINE)


def parse_trace(path: Path):
    text = path.read_text()

    # user turns: lines starting with "> " right after **User**
    user_msgs = []
    for block in USER_BLOCK_RE.finditer(text):
        raw = block.group(1)
        # a user turn may span multiple "> " quoted lines
        quoted_lines = [
            l.strip().lstrip(">").strip()
            for l in raw.splitlines()
            if l.strip().startswith(">")
        ]
        if quoted_lines:
            user_msgs.append(" ".join(quoted_lines))
        else:
            user_msgs.append(raw.strip())

    # find the LAST markdown table in the file (final agreed shortlist)
    table_blocks = re.findall(
        r"\|\s*#\s*\|\s*Name\s*\|.*?\n((?:\|.*\n?)+)", text
    )
    expected_names = []
    if table_blocks:
        last_table = table_blocks[-1]
        for row in TABLE_ROW_RE.finditer(last_table):
            name = row.group(1).strip()
            expected_names.append(name)

    return {
        "file": path.name,
        "user_turns": user_msgs,
        "expected_shortlist": expected_names,
    }


def load_all_traces():
    return [parse_trace(p) for p in sorted(TRACES_DIR.glob("*.md"))]


if __name__ == "__main__":
    traces = load_all_traces()
    for t in traces:
        print(f"{t['file']}: {len(t['user_turns'])} user turns, "
              f"expected shortlist ({len(t['expected_shortlist'])}): {t['expected_shortlist']}")
