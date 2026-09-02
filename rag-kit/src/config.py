"""租户级配置。

设计目标：一套代码服务多个客户，差异全部收敛到 config/tenants/*.yaml，
新增客户不改代码。这是把边际交付时间从 60 小时压到 20 小时的前提。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # 环境退化（PyYAML 缺失）时的零依赖回退
    yaml = None


@dataclass
class ParserConfig:
    """文档解析。RAG 效果的天花板在这里，80% 的效果问题不怪模型，怪解析。"""

    engine: str = "docling"  # docling | pymupdf
    ocr: bool = False  # 扫描件是否需要 OCR
    ocr_lang: str = "ch"
    drop_header_footer: bool = True  # 去掉页眉页脚、目录页码
    table_mode: str = "kv_pairs"  # kv_pairs | markdown | skip
    min_text_len: int = 30  # 短于此长度的页面视为解析失败


@dataclass
class ChunkConfig:
    """切片策略。父子 chunk 是性价比最高的默认选择。"""

    strategy: str = "parent_child"  # parent_child | recursive | fixed
    child_size: int = 128  # 用于精确定位
    parent_size: int = 768  # 用于给 LLM 完整上下文
    overlap: int = 32
    respect_headings: bool = True  # 按标题层级切，不按字符硬切


@dataclass
class RetrievalConfig:
    """检索配置。纯向量是不够的，必须加关键词路和重排。"""

    dense_model: str = "BAAI/bge-m3"
    dense_dim: int = 1024
    sparse: bool = True  # BM25 关键词路，对型号/代号/人名很关键
    top_k_coarse: int = 50
    top_k_final: int = 8
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    query_rewrite: bool = True


@dataclass
class GuardConfig:
    """生成护栏。合规场景能否上线，主要看这一层。"""

    require_citation: bool = True  # 强制引用溯源
    refuse_threshold: float = 0.35  # 最高检索分低于此值即拒答
    max_context_chars: int = 12000
    injection_check: bool = True  # Prompt 注入检测


@dataclass
class SecurityConfig:
    """权限与审计。这是样例 B 的核心，也是相对平台竞争者的护城河。"""

    acl_enabled: bool = False
    default_level: str = "public"  # public | internal | restricted
    audit_enabled: bool = True
    audit_path: str = "./data/{tenant}/audit.jsonl"
    # 配置映射演示用：文档源(stem) -> 密级。预留与甲方 IAM/钉钉/企微角色对接点。
    acl_map: dict = field(default_factory=dict)


@dataclass
class LLMConfig:
    """生成模型。支持 ollama / vllm / openai 三种后端，靠配置切换。"""

    provider: str = "ollama"  # ollama | vllm | openai
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen3:14b"
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class TenantConfig:
    tenant: str = "default"
    data_dir: str = "./data/default"
    collection: str = "kb_default"
    qdrant_url: str = "http://localhost:6333"
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @property
    def audit_path(self) -> Path:
        path = Path(self.security.audit_path.format(tenant=self.tenant))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def parsed_dir(self) -> Path:
        return Path(self.data_dir) / "parsed"


def _merge(cls, data: dict | None):
    if not data:
        return cls()
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _mini_yaml_load(text: str) -> dict:
    """极简 YAML 解析器：支持 2 层嵌套映射 + 标量（str/int/float/bool/null）。

    仅用于环境退化、PyYAML 缺失时的回退；覆盖本项目所有 configs 的结构。
    """

    def parse_scalar(s: str):
        if s == "":
            return None
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        low = s.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("null", "~", "none"):
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    root: dict = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if "#" in line:
            idx = line.find("#")
            if idx == 0 or line[idx - 1].isspace():
                line = line[:idx]
        line = line.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_part = line.strip()
        if ":" not in key_part:
            continue
        ci = key_part.find(":")
        key = key_part[:ci].strip()
        rest = key_part[ci + 1:].strip()
        value = parse_scalar(rest) if rest != "" else None
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value is None:
            new_node: dict = {}
            parent[key] = new_node
            stack.append((indent, new_node))
        else:
            parent[key] = value
    return root


def load_config(path: str | Path) -> TenantConfig:
    """从 YAML 加载租户配置（PyYAML 缺失时回退到内置极简解析器）。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        raw = yaml.safe_load(text) or {}
    else:
        raw = _mini_yaml_load(text) or {}

    cfg = TenantConfig(
        tenant=raw.get("tenant", "default"),
        data_dir=raw.get("data_dir", f"./data/{raw.get('tenant', 'default')}"),
        collection=raw.get("collection", f"kb_{raw.get('tenant', 'default')}"),
        qdrant_url=raw.get("qdrant_url", os.getenv("QDRANT_URL", "http://localhost:6333")),
        parser=_merge(ParserConfig, raw.get("parser")),
        chunk=_merge(ChunkConfig, raw.get("chunk")),
        retrieval=_merge(RetrievalConfig, raw.get("retrieval")),
        guard=_merge(GuardConfig, raw.get("guard")),
        security=_merge(SecurityConfig, raw.get("security")),
        llm=_merge(LLMConfig, raw.get("llm")),
    )
    return cfg
