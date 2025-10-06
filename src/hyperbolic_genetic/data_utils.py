import re

def _extract_glottocode(label: str | None) -> str | None:
    """
    Extracts a glottocode (4 letters, 4 digits) from a string.
    e.g., 'Bogia [monu1249]' -> 'monu1249'
    """
    if label is None:
        return None
    match = re.search(r'\[([a-z]{4}\d{4})\]', label)
    if match:
        return match.group(1)
    return None

def _parse_label(s: str, i: int):
    """
    Read a Newick node label starting at position *i*.
    Returns (label_or_None, next_index).
    Handles doubled apostrophes inside quoted labels.
    """
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    if i >= n:
        return None, i

    label = ""
    if s[i] == "'":
        i += 1
        while i < n:
            ch = s[i]
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    label += "'"
                    i += 2
                else:
                    i += 1
                    break
            else:
                label += ch
                i += 1
    else:
        while i < n and s[i] not in ",:() ;":
            label += s[i]
            i += 1
        label = label.strip()

    if i < n and s[i] == ":":
        i += 1
        while i < n and s[i] not in ",);":
            i += 1

    return (label if label else None), i


class _Node:
    __slots__ = ("label", "children")

    def __init__(self, label=None):
        self.label = label
        self.children = []


def _parse_subtree(s: str, i: int):
    if s[i] == "(":
        node = _Node()
        i += 1
        while True:
            child, i = _parse_subtree(s, i)
            node.children.append(child)
            if s[i] == ",":
                i += 1
                continue
            elif s[i] == ")":
                i += 1
                break
        label, i = _parse_label(s, i)
        if label is not None:
            node.label = label
        return node, i
    else:
        label, i = _parse_label(s, i)
        return _Node(label), i


def _parse_newick(s: str):
    s = s.strip()
    if not s.endswith(";"):
        raise ValueError("Newick string must end with ';'")
    root, _ = _parse_subtree(s, 0)
    return root


def _build_adj(node: _Node, adj: dict[str, list[str]]):
    parent_glottocode = _extract_glottocode(node.label)
    if parent_glottocode is not None and parent_glottocode not in adj:
        adj[parent_glottocode] = []

    for child in node.children:
        child_glottocode = _extract_glottocode(child.label)
        if child_glottocode is not None:
            if child_glottocode not in adj:
                adj[child_glottocode] = []
            if (parent_glottocode is not None
                    and parent_glottocode != child_glottocode
                    and child_glottocode not in adj[parent_glottocode]):
                adj[parent_glottocode].append(child_glottocode)
        _build_adj(child, adj)


def newick_to_adjacency_list(input):
    lines = input.split("\n")
    master_adj: dict[str, list[str]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        tree = _parse_newick(line)
        local = {}
        _build_adj(tree, local)
        for k, children in local.items():
            if k not in master_adj:
                master_adj[k] = []
            for c in children:
                if c not in master_adj[k]:
                    master_adj[k].append(c)
    return master_adj
