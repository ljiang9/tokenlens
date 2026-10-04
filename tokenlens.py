#!/usr/bin/env python3
"""tokenlens — 在调用 API 之前，先估算 token 用量和费用。

完全离线、只用 Python 标准库。token 估算是启发式的（见 count_tokens
的文档），不要把它当成 tiktoken 的精确值；价格表是近似值，使用前请核对官网。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# 价格表：INPUT $ / 1M tokens。近似值，2026-10 整理，使用前请以官网为准。
# 想加模型 / 改价格：直接改这个 dict 即可。
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-4o":        {"input_per_1m": 2.50,  "context": 128_000, "note": "OpenAI"},
    "gpt-4o-mini":   {"input_per_1m": 0.15,  "context": 128_000, "note": "OpenAI"},
    "claude-sonnet": {"input_per_1m": 2.00,  "context": 200_000, "note": "Anthropic Sonnet 5.5"},
    "claude-haiku":  {"input_per_1m": 1.00,  "context": 200_000, "note": "Anthropic Haiku 4.5"},
    "deepseek":      {"input_per_1m": 0.435, "context": 128_000, "note": "DeepSeek V4 Pro"},
    "qwen":          {"input_per_1m": 0.40,  "context": 131_072, "note": "Qwen3.7 Plus"},
}
PRICES_AS_OF = "2026-10（近似值，请以官网为准）"

# ---------------------------------------------------------------------------
# 启发式分词器（诚实说明）
#
# 方法：
#   tokens ≈ CJK字符数 × 1
#           + Σ ASCII单词 ceil(单词长度 / 4)
#           + ASCII标点每字符 1
#           + 其他非ASCII字符（emoji等）每字符 2
#   空白字符不单独计数（BPE 通常把空格并入相邻 token，会略微低估）。
#
# 误差范围（与 tiktoken/cl100k 类分词器对比的经验值）：
#   英文散文 ±15%、代码 ±20%、中日韩文本 ±20%、emoji 密集文本可能低估更多。
# 结论：只适合做"量级判断"（选模型、估预算、判断是否超窗），不适合精确计费。
# ---------------------------------------------------------------------------
_CJK_RE = re.compile(
    r"["
    r"\u3400-\u4dbf"          # CJK Ext A
    r"\u4e00-\u9fff"          # CJK Unified
    r"\uf900-\ufaff"          # CJK Compatibility
    r"\U00020000-\U0002a6df"  # CJK Ext B
    r"\u3040-\u30ff"          # Hiragana + Katakana
    r"\uac00-\ud7af"          # Hangul syllables
    r"\u1100-\u11ff\u3130-\u318f"  # Hangul Jamo / Compatibility Jamo
    r"\uff00-\uffef"          # 全角字符（含全角标点）
    r"\u3000-\u303f"          # CJK 标点符号
    r"]"
)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_WS_RE = re.compile(r"\s+")


def count_tokens(text: str) -> int:
    """启发式估算 token 数。详见模块顶部的方法与误差说明。"""
    cjk = len(_CJK_RE.findall(text))
    words = _WORD_RE.findall(text)
    word_tokens = sum((len(w) + 3) // 4 for w in words)  # ceil(len/4)
    rest = _CJK_RE.sub("", text)
    rest = _WORD_RE.sub("", rest)
    rest = _WS_RE.sub("", rest)
    other = sum(2 if ord(ch) > 127 else 1 for ch in rest)
    return cjk + word_tokens + other


def cost_for(tokens: int, price_per_1m: float) -> float:
    return tokens / 1_000_000 * price_per_1m


def fmt_money(d: float) -> str:
    if d < 0.01:
        return f"${d:.6f}"
    return f"${d:,.4f}"


def resolve_model(name: str) -> tuple[str, dict]:
    if name in MODELS:
        return name, MODELS[name]
    # 允许大小写/下划线等宽松匹配
    key = name.strip().lower().replace("_", "-")
    for k, v in MODELS.items():
        if k.lower() == key:
            return k, v
    close = ", ".join(sorted(MODELS))
    raise SystemExit(f"error: 未知模型 '{name}'，可用：{close}")


# ---------------------------------------------------------------------------
# 目录扫描
# ---------------------------------------------------------------------------
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv",
             ".tox", "dist", "build", ".idea", ".vscode", "target", ".hg", ".svn"}
BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf",
               ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".pyc", ".pyo",
               ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".mp4", ".mp3",
               ".wav", ".ogg", ".woff", ".woff2", ".ttf", ".eot", ".otf",
               ".sqlite", ".db", ".parquet", ".onnx", ".bin", ".pth",
               ".h5", ".hdf5", ".wasm", ".class", ".jar", ".psd", ".ai",
               ".dmg", ".iso", ".pkl", ".npy", ".npz", ".lockb"}
MAX_READ = 2_000_000  # 单个文件最多读 2MB 做估算


def load_gitignore_patterns(root: str) -> list[str]:
    pats: list[str] = []
    p = os.path.join(root, ".gitignore")
    if os.path.isfile(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("!"):
                    pats.append(line)
    return pats


def gitignore_match(rel: str, name: str, pats: list[str]) -> bool:
    for pat in pats:
        p = pat.strip().lstrip("/")
        if p.endswith("/"):
            if rel == p[:-1] or rel.startswith(p):
                return True
        elif p.startswith("*."):
            if name.endswith(p[1:]):
                return True
        elif "*" in p:
            if re.fullmatch(re.escape(p).replace(r"\*", ".*"), name):
                return True
        elif p == name or rel == p or rel.startswith(p + "/"):
            return True
    return False


def is_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(8192)
    except OSError:
        return True


def scan_dir(root: str) -> tuple[list[dict], dict]:
    root = os.path.abspath(root)
    pats = load_gitignore_patterns(root)
    files: list[dict] = []
    skipped = {"binary": 0, "gitignored": 0, "unreadable": 0}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS
                             and not gitignore_match(os.path.relpath(os.path.join(dirpath, d), root), d, pats))
        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, root)
            if gitignore_match(rel, fn, pats):
                skipped["gitignored"] += 1
                continue
            if os.path.splitext(fn)[1].lower() in BINARY_EXTS or is_binary(fp):
                skipped["binary"] += 1
                continue
            try:
                with open(fp, encoding="utf-8", errors="replace") as f:
                    text = f.read(MAX_READ + 1)
                truncated = len(text) > MAX_READ
                text = text[:MAX_READ]
            except OSError:
                skipped["unreadable"] += 1
                continue
            files.append({"path": rel, "chars": len(text),
                          "tokens": count_tokens(text), "truncated": truncated})
    total = sum(f["tokens"] for f in files)
    return files, {"total_tokens": total, "files": len(files), "skipped": skipped}


# ---------------------------------------------------------------------------
# --fit / --split
# ---------------------------------------------------------------------------
def fit_report(tokens: int, model_name: str) -> dict:
    name, m = resolve_model(model_name)
    ctx = m["context"]
    headroom = ctx - tokens
    return {"model": name, "context": ctx, "tokens": tokens,
            "fits": headroom >= 0, "headroom": headroom,
            "headroom_pct": round(headroom / ctx * 100, 2) if ctx else 0.0}


def split_plan(tokens: int, n: int, overlap_ratio: float = 0.10,
               overlap_cap: int = 200) -> dict:
    if n < 1:
        raise SystemExit("error: --split 需要 n >= 1")
    chunk = (tokens + n - 1) // n
    overlap = min(overlap_cap, max(1, int(chunk * overlap_ratio)))
    stride = max(1, chunk - overlap)
    return {"chunks": n, "chunk_tokens": chunk, "overlap_tokens": overlap,
            "stride_tokens": stride,
            "note": "按 token 数均分；实际切分请按段落/语义边界微调，"
                    "overlap 用于保留跨块上下文"}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def cost_table(tokens: int, only: str | None = None) -> list[tuple[str, float, float]]:
    rows = []
    names = [only] if only else sorted(MODELS, key=lambda k: MODELS[k]["input_per_1m"])
    for name in names:
        _, m = resolve_model(name)
        rows.append((name, m["input_per_1m"], cost_for(tokens, m["input_per_1m"])))
    return rows


def print_prompt_text(path: str, chars: int, tokens: int, args) -> int:
    """返回 exit code。"""
    print(f"Prompt: {path}（{chars:,} 字符）")
    print(f"估算 tokens：约 {tokens:,}（启发式 ±20%，非 tiktoken 精确值）\n")
    if args.model:
        name, m = resolve_model(args.model)
        print(f"模型 {name}：{fmt_money(cost_for(tokens, m['input_per_1m']))} "
              f"（单价 ${m['input_per_1m']}/1M，价格{PRICES_AS_OF}）")
    else:
        print(f"各模型费用估算（input 单价，价格{PRICES_AS_OF}）：")
        print(f"  {'模型':<14}{'$/1M':>10}{'预估费用':>14}")
        for name, price, c in cost_table(tokens):
            print(f"  {name:<14}{('$%.3f' % price):>10}{fmt_money(c):>14}")
    exit_code = 0
    if args.fit:
        r = fit_report(tokens, args.fit)
        print(f"\n--fit {r['model']}（context {r['context']:,}）：")
        if r["fits"]:
            print(f"  放得下 ✓  剩余 {r['headroom']:,} tokens（{r['headroom_pct']}% 空闲）")
            print(f"  建议给输出预留 ≥ {min(r['headroom'], 4096):,} tokens")
        else:
            print(f"  放不下 ✗  超出 {-r['headroom']:,} tokens")
            exit_code = 2
    if args.split:
        p = split_plan(tokens, args.split)
        print(f"\n--split 切分建议（{p['chunks']} 块）：")
        print(f"  每块约 {p['chunk_tokens']:,} tokens，块间重叠 {p['overlap_tokens']:,} tokens，"
              f"步长 {p['stride_tokens']:,} tokens")
        print(f"  {p['note']}")
        if args.fit:
            _, m = resolve_model(args.fit)
            if p["chunk_tokens"] <= m["context"]:
                print(f"  每块可放入 {args.fit}（context {m['context']:,}）✓")
            else:
                print(f"  每块仍超出 {args.fit} 的 context，需要更多块 ✗")
                exit_code = 2
    if args.budget is not None:
        cheapest = min(cost_for(tokens, m["input_per_1m"]) for m in MODELS.values())
        if cheapest > args.budget:
            print(f"\n⚠ 预算警告：最便宜的方案约 {fmt_money(cheapest)}，"
                  f"超过预算 ${args.budget:.2f}", file=sys.stderr)
            exit_code = 2
        else:
            print(f"\n预算 OK：最便宜约 {fmt_money(cheapest)} ≤ ${args.budget:.2f}")
    return exit_code


def print_dir_text(root: str, files: list[dict], summary: dict) -> None:
    print(f"目录：{root}")
    print(f"文本文件 {summary['files']} 个，估算 tokens 合计约 {summary['total_tokens']:,}")
    sk = summary["skipped"]
    if any(sk.values()):
        print(f"跳过：二进制 {sk['binary']}，.gitignore {sk['gitignored']}，"
              f"不可读 {sk['unreadable']}")
    if files:
        print(f"\n{'tokens':>10}  文件")
        for f in sorted(files, key=lambda x: -x["tokens"])[:10]:
            mark = "（截断>2MB）" if f["truncated"] else ""
            print(f"{f['tokens']:>10,}  {f['path']}{mark}")
        if len(files) > 10:
            print(f"  … 另有 {len(files) - 10} 个文件未列出")
    print(f"\n费用估算（input 单价，价格{PRICES_AS_OF}）：")
    print(f"  {'模型':<14}{'$/1M':>10}{'预估费用':>14}")
    for name, price, c in cost_table(summary["total_tokens"]):
        print(f"  {name:<14}{('$%.3f' % price):>10}{fmt_money(c):>14}")
    print("\n注：token 为启发式估算（±20%），非精确计费口径。")


def build_json(args, **kw) -> dict:
    out: dict = {"tool": "tokenlens", "version": __version__,
                 "method": "heuristic (±20%, not tiktoken)",
                 "prices_as_of": PRICES_AS_OF}
    out.update(kw)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tokenlens",
        description="调用 API 之前，先估算 token 用量和费用（离线、启发式）。")
    ap.add_argument("prompt", nargs="?", help="prompt 文件路径")
    ap.add_argument("--dir", metavar="DIR", help="扫描代码目录，按文件统计 tokens")
    ap.add_argument("--model", metavar="NAME", help="只算一个模型的费用")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--budget", type=float, metavar="DOLLARS",
                    help="预算上限；最便宜方案超预算则警告并 exit 2")
    ap.add_argument("--fit", metavar="MODEL",
                    help="检查 prompt 是否放得进该模型的 context window")
    ap.add_argument("--split", type=int, metavar="N",
                    help="给出切成 N 块的 chunking 建议")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    if args.dir and args.prompt:
        ap.error("--dir 不能与 prompt 文件同时使用")
    if (args.fit or args.split) and not args.prompt:
        ap.error("--fit/--split 需要指定一个 prompt 文件")
    if args.model:
        resolve_model(args.model)  # 提前校验名字
    if args.fit:
        resolve_model(args.fit)

    # ---- 目录模式 ----
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"error: 目录不存在：{args.dir}", file=sys.stderr)
            return 1
        files, summary = scan_dir(args.dir)
        if args.json:
            costs = {n: cost_for(summary["total_tokens"], p)
                     for n, p, _ in cost_table(summary["total_tokens"])}
            print(json.dumps(build_json(args, mode="dir", root=os.path.abspath(args.dir),
                                        files=files, summary=summary,
                                        costs=costs),
                             ensure_ascii=False, indent=2))
        else:
            print_dir_text(args.dir, files, summary)
        return 0

    # ---- prompt 文件模式 ----
    if not args.prompt:
        ap.error("需要指定 prompt 文件，或用 --dir 扫描目录")
    if not os.path.isfile(args.prompt):
        print(f"error: 文件不存在：{args.prompt}", file=sys.stderr)
        return 1
    with open(args.prompt, encoding="utf-8", errors="replace") as f:
        text = f.read()
    chars, tokens = len(text), count_tokens(text)

    if args.json:
        payload: dict = {"mode": "prompt", "file": args.prompt,
                         "chars": chars, "tokens": tokens}
        if args.model:
            name, m = resolve_model(args.model)
            payload["costs"] = {name: cost_for(tokens, m["input_per_1m"])}
        else:
            payload["costs"] = {n: cost_for(tokens, p)
                                for n, p, _ in cost_table(tokens)}
        if args.fit:
            payload["fit"] = fit_report(tokens, args.fit)
        if args.split:
            payload["split"] = split_plan(tokens, args.split)
        if args.budget is not None:
            cheapest = min(payload["costs"].values()) if payload["costs"] else 0
            payload["budget"] = {"limit": args.budget, "cheapest": cheapest,
                                 "ok": cheapest <= args.budget}
        print(json.dumps(build_json(args, **payload), ensure_ascii=False, indent=2))
        over = args.budget is not None and not payload["budget"]["ok"]
        nofit = args.fit and not payload["fit"]["fits"]
        return 2 if (over or nofit) else 0

    return print_prompt_text(args.prompt, chars, tokens, args)


if __name__ == "__main__":
    sys.exit(main())
