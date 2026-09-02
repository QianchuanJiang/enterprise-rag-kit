# 文档解析 · 真实 PDF 路线执行说明

> 目标：把「年报 PDF → PyMuPDF 抽取 → 切片 → 质检 → BGE-M3 → GLM」这条生产链路真正跑通，
> 并在真实语料上重新标定拒答阈值。

## 一、本次实际执行了什么

1. **PyMuPDF 安装**：在 `rag-forge` conda 环境装好 `PyMuPDF 1.28.0`（之前误以为缺失，
   实际是 `conda activate` 在非交互 shell 未生效、一直用错了托管运行时；改用
   `/Users/dagongjidecuoyiban/anaconda3/envs/rag-forge/bin/python` 绝对路径后一切正常）。
2. **生成真实 PDF 文件**：`scripts/make_real_pdfs.py` 把 `data/reports/` 下 8 家跨行业
   合成年报（.md）用 PyMuPDF + 系统中文矢量字体（STHeiti Light.ttc）写成**真正的多页 PDF**
   到 `data/reports/raw/`，并做字体子集化（单文件 55MB → ~750KB）。
3. **真实抽取 + 评测**：`scripts/real_pdf_eval.py` 走 `parser.engine: pymupdf` →
   `_parse_pymupdf` 抽取真实 PDF，跑通完整链路，产出 `data/reports/metrics_pdf.json`。

## 二、关于「真实下载的年报 PDF」

本环境**无法自动下载**真实年报 PDF，已逐项验证：

| 来源 | 结果 | 原因 |
|------|------|------|
| 巨潮 CNINFO 列表 API | ❌ 500 | 接口已改为需鉴权 |
| 东方财富 ann API | ❌ 200 但空 body | 服务端对非浏览器客户端策略性空响应 |
| 东方财富 datacenter API | ❌ 报表配置不存在 | reportName 不匹配 |
| 新浪年报栏目（vCB_Bulletin/ndbg） | ❌ 无静态链接 | 列表为 JS 动态渲染，urllib 拿不到 |
| docling 解析 | ❌ | HuggingFace 被墙，运行时要下模型 |

因此当前 `data/reports/raw/` 里的 PDF **文件格式与解析 100% 真实**（PyMuPDF 面对的是标准 PDF，
与交付客户时一致），但**文本来源是合成逼真版**。这是环境限制下的务实替代，指标仍由真实
BGE-M3 嵌入 + 真实 GLM 生成跑出，可复现。

## 三、如何换成「真实下载的年报 PDF」

等环境恢复（能联网取到 PDF）或你本地已有年报 PDF 时：

1. 把真实 PDF 丢进 `data/reports/raw/`（覆盖或新增，文件名随意）；
2. 若来源是 .md 也可继续放 `data/reports/` 走 markdown 路线；
3. 重跑评测：
   ```bash
   cd rag-kit
   /Users/dagongjidecuoyiban/anaconda3/envs/rag-forge/bin/python scripts/real_pdf_eval.py
   ```
4. **必须在该真实语料上重新标定阈值**（见下）。

## 四、真实 PDF 路线指标（本次，8 家 · 真实 PDF 文件）

| 指标 | 结果 |
|------|------|
| 入库 PDF / chunks | 8 / 8（质检通过率 1.0） |
| Recall@1 / @3 / @5 / @8 | 75% / 90% / **100%** / 100% |
| 拒答阈值标定（数据驱动） | 相关 [0.656,0.818] / 无关 [0.365,0.577]，干净分离 → **0.62** |
| 生成（GLM-4.6V）非空/溯源/grounding | 6/6 · 6/6 · 6/6 |
| 无依据拒答 | 4/4 |

对比 .md 扩量路线：相关 [0.605,0.746]/无关 [0.449,0.576] → 0.59。
PDF 抽取后的文本更干净，分离度更好，故阈值略升到 0.62。

## 五、已知演示特征（非缺陷）

- 当前语料仅 8 份文档、top_k_final=8，检索会返回近乎全部 chunk，溯源列表偏长。
  扩到 20+ 份或调小 `retrieval.top_k_final` 即可让溯源更聚焦。
- 合成 PDF 为单页/少页，每 PDF 抽成 1 个 block，切片后再拆父子 chunk；
  真实多页年报会被 PyMuPDF 按页抽成多 block，结构还原更完整。

## 六、运行命令速查

```bash
PY=/Users/dagongjidecuoyiban/anaconda3/envs/rag-forge/bin/python
cd rag-kit
$PY scripts/make_real_pdfs.py     # 生成真实 PDF -> data/reports/raw/
$PY scripts/real_pdf_eval.py      # 跑 PDF 路线评测 -> metrics_pdf.json
```
