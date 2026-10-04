# tokenlens 🔍

**调用 API 之前，先看看这句话值多少钱。**

tokenlens 是一个离线的 token 用量 / 费用估算小工具：把 prompt 文件或整个代码目录丢给它，
它告诉你大概多少 tokens、各家模型分别要花多少钱、能不能塞进 context window、超了怎么切块。

- 零依赖：只用 Python 标准库（Python 3.10+），`pip install` 什么都不用
- 完全离线：不联网、不调任何 API，价格表是内置的
- 诚实声明：token 数是**启发式估算（误差约 ±20%）**，不是 tiktoken 精确值——适合做量级判断，不适合精确计费

## 快速开始

```bash
# 估算一个 prompt 文件的 tokens + 各模型费用
python -m tokenlens prompt.txt

# 扫描整个代码目录
python -m tokenlens --dir ./src

# 只算一个模型
python -m tokenlens prompt.txt --model deepseek

# 输出 JSON（给脚本用）
python -m tokenlens prompt.txt --json

# 预算检查：最便宜的方案超 $0.50 就警告
python -m tokenlens prompt.txt --budget 0.50

# 这个 prompt 放得进 gpt-4o 的窗口吗？
python -m tokenlens prompt.txt --fit gpt-4o

# 放不进？给出切 4 块的方案
python -m tokenlens prompt.txt --fit gpt-4o --split 4
```

也可以直接运行：`python tokenlens.py prompt.txt`（在项目目录内）。

## 示例输出

```
Prompt: examples/sample-prompt.txt（486 字符）
估算 tokens：约 132（启发式 ±20%，非 tiktoken 精确值）

各模型费用估算（input 单价，价格2026-10（近似值，请以官网为准））：
  模型              $/1M        预估费用
  qwen             $0.400     $0.000053
  deepseek         $0.435     $0.000057
  gpt-4o-mini      $0.150     $0.000020
  ...
```

## 估算方法（诚实版）

```
tokens ≈ CJK 字符数 × 1
       + Σ ASCII 单词 ceil(单词长度 / 4)
       + ASCII 标点每字符 1
       + 其他非 ASCII 字符（emoji 等）每字符 2
```

- 空白字符不单独计数（BPE 通常把空格并入相邻 token，会略微低估）。
- 经验误差：英文散文 ±15%、代码 ±20%、中日韩文本 ±20%、emoji 密集文本可能低估更多。
- **不要用它做精确计费对账**，只做"选模型、估预算、判断超窗"的量级判断。

## 价格表

内置在 `tokenlens.py` 顶部的 `MODELS` 字典里（input $/1M tokens + context window），
想加模型、改价格直接改那个 dict。当前为 **2026-10 整理的近似值**，实际扣费以各家官网为准。

## 目录扫描规则

`--dir` 会跳过：`.git`、`__pycache__`、`node_modules`、`.venv` 等常见目录；
按扩展名跳过二进制文件（图片、压缩包、`.pyc`、音视频、模型权重等），
外加内容检测（含 `\x00` 字节即判为二进制）。根目录的 `.gitignore` 会被简单解析
（支持 `*.ext`、`dir/`、精确名），复杂规则是 best-effort。

## 已知 limitation

- token 估算是启发式的，不是各家官方分词器；不同模型的实际分词结果本就不同。
- 只算 **input** 费用，不估 output（输出长度你控制不了，工具也猜不到）。
- 价格是静态表，不会自动更新；涨价了记得手动改 `MODELS`。
- `--split` 只给"按 token 数均分"的数学方案，实际切分请按段落/语义边界微调。

## License

MIT © 2026 ljiang9
