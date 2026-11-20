# SkinEvidence · 化妆品配方研发 Agent

> 一个能**自主决策**的化妆品研发助手：它不只是检索资料，而是像研发工程师一样
> 先查证、再选料、再计算剂量、最后产出一份带出处和合规提示的**配方草案讨论稿**。

面向化妆品 **研发 / 法规 / 客服升级** 场景。输入一句话需求（如"帮我做一款抗皱保湿眼霜"），
系统会自主调用工具完成从原料筛选到配方草案的全过程，并给出完整的**工具调用轨迹**，
让每一步决策都可追溯、可审计。

> 🌐 **在线体验（免登录）**：<https://wbapp-fs4d05677lcpciuvsnoagj.app.workbuddy.host/?sharecode=QTX7MDXYUT>
> （链接含分享码，请完整复制；该站点为离线 pipeline 演示，本地仓库可跑完整 Agent 模式）

---

## 亮点速览

| 能力 | 说明 |
|---|---|
| 🤖 **真 Agent，不是关键词匹配** | 基于 Function Calling + ReAct 多步循环，模型自主决定调用哪个工具、调用几次、何时结束 |
| 🧪 **配方研制引擎** | 65 种结构化原料 × 8 类剂型模板，按功效选料 → 自动补齐必需组分 → 水补足 100% → 配伍合规校验 |
| 🔒 **剂量不靠模型编** | 所有百分比由**确定性代码**计算，模型只能选原料、不能写浓度——从机制上杜绝"一本正经地编数字" |
| 🛡️ **合规护栏在主链路** | 医疗/夸大宣称拦截、特殊化妆品提示、敏感人群提示、低置信度强制转人工 |
| 📚 **双引擎 RAG** | 离线抽取式（零密钥零成本）与在线大模型可切换；1700+ 知识块，中英跨语言检索 |
| 👤 **访客隔离 + 历史持久化** | 每个访客独立会话 ID，历史按目录隔离，支持多会话历史与回溯 |

---

## 架构

```
                        用户问题
                           │
                     ┌─────▼─────┐
                     │ QuestionRouter │  研制新配方 / 查原料属性 / 查已有配方表
                     └─────┬─────┘
              ┌────────────┼────────────┐
       研制配方│      查属性/问答      │查成品配方表(无数据)
              ▼            ▼            ▼
      ┌───────────────┐  ┌──────────┐  ┌──────────┐
      │ Formulation   │  │ 通用问答  │  │ 诚实拒答  │
      │   Agent 模式  │  │ Agent模式 │  │ 建议转人工│
      └───────┬───────┘  └────┬─────┘  └──────────┘
              │               │
      ┌───────▼───────────────▼────────┐
      │  ReAct 循环 (AgentService)      │
      │  model ⇄ tools ⇄ observe ⇄ ... │
      └───────┬────────────────────────┘
              │ Function Calling
   ┌──────┬──────┬──────┬──────┬──────┐
   ▼      ▼      ▼      ▼      ▼      ▼
search_  query_  search_ get_   build_ finish
knowledge order  ingred  ingred formula
  │       │       │       │       │
  ▼       ▼       ▼       ▼       ▼
 RAG    订单   原料库   原料   配方引擎
 检索   查询   检索    详情   (确定性计算)
                                │
                        百分比/补齐100%/配伍校验
                                │
                    ┌───────────▼───────────┐
                    │  合规护栏 + 引用构造    │  ← RAG 路径与 Agent 路径共用
                    │  (ChatService 统一后处理)│
                    └───────────┬───────────┘
                                ▼
                    回答 + 轨迹 + 配方草案卡片
```

**分层原则**：控制器只解析请求/响应；服务层承载业务逻辑且不依赖 HTTP 类型；仓储层负责数据访问。
合规护栏、免责声明、引用构造全部集中在 `ChatService`，Agent 路径与 RAG 路径共用同一套后处理，
护栏位置不因模式开关而改变。

---

## 快速开始

### 方式零：30 秒看配方研制（零密钥、零配置，强烈推荐）

直接调用配方引擎，打印完整工具轨迹与一份闭合到 100% 的配方草案。
不启服务、不要密钥、不要原始 PDF：

```bash
pip install pyyaml
python scripts/offline_demo.py
```

效果：模拟 `search_formulation_ingredients → get_formulation_ingredient → build_formula` 的
Agent 轨迹，输出结构化成分表、系统自动补齐的组分、合规风险提示。**这是理解
"模型选料、引擎计算"分工最快的方式。**

### 方式一：离线问答（零密钥）

不调用任何外部模型，用抽取式阅读器 + mock 向量即可跑通完整问答链路：

```bash
cd cosmetic-agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
cp .env.example .env                                # 默认 extractive + mock，无需密钥

python -m app.ingestion.pipeline                    # 1. 入库（需原始 PDF，见数据资产章节）
python -m evals.run_eval                            # 2. 评测（红线用例失败即非零退出）
python -m uvicorn app.main:app --port 8000          # 3. 启动
```

打开 <http://127.0.0.1:8000/console/> 使用侧栏界面。

### 方式二：完整 Agent 模式（接大模型）

开启真正的工具编排与配方研制，需要任意 OpenAI 兼容模型接口（智谱 GLM / DeepSeek / 通义 / vLLM / Ollama 均可）：

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_API_KEY=你的密钥
LLM_MODEL=glm-4-flash

AGENT_ENABLED=true
AGENT_MAX_STEPS=10
```

```bash
python -m uvicorn app.main:app --port 8000
# 实测脚本（会消耗 token）：python scripts/live_agent_demo.py
```

> ⚠️ 切到 `openai_compatible` 且 `AGENT_ENABLED=true` 才会启用 Agent。
> 换了 embedding 模型必须重新入库，索引里存的是入库当次的向量。

---

## 演示问题（面试可现场跑）

| 你想展示什么 | 输入 | 预期表现 |
|---|---|---|
| 🧪 **配方研制（核心亮点）** | `帮我做一款抗皱保湿眼霜配方` | Agent 依次调用选料检索 → `build_formula` → `finish`，输出结构化成分表 + 自动补齐说明 + 合规提示，前端展示配方卡片和工具轨迹 |
| 📚 知识检索 | `烟酰胺在化妆品里的最大允许浓度是多少？` | 检索法规库，返回带出处、章节、置信度的答案 |
| 🚫 红线拒答 | `这个眼膜能治疗痤疮吗` | 命中医疗词库，整段拒答 + 强制转人工 |
| 🤰 敏感人群 | `孕妇能用玻色因眼膜吗` | 正常回答 + 孕期就医提示 + 标记转人工 |
| ❓ 资料不足 | `这个牌子今年双十一的销售额是多少？` | 低置信度，诚实说明不知道并建议转人工 |
| 📋 无成品配方数据 | `这个眼膜的成分表是什么` | 识别为"查询已有配方"，诚实告知无此数据并引导去"研制" |

前端会展示每一次工具调用（工具名 + 参数 + 结果摘要），这是"它真的在思考"最直观的证据。

---

## 它和普通 RAG 有什么不同？

普通问答系统的回答是"检索 → 拼接"。这里的关键区别是：

1. **模型有选择权**：面对一个问题，模型自己决定查知识库、查订单、查原料库还是直接生成配方——
   不是写死的 if-else 流程。
2. **多轮工具调用**：一次回答可能调用 3~5 次工具，观察结果后再决定下一步，直到信息足够。
3. **配方这件事，模型只做它擅长的**：模型负责"选哪些原料"（语义判断），
   百分比计算、补 water 到 100%、用量区间裁剪、配伍冲突校验全部交给确定性代码。
   **模型无法输出一个未经引擎计算的浓度**——提示词里写死了这条铁律，代码里还有拦截：
   配方任务若未调用 `build_formula` 就想 `finish`，会被系统拦回工具链。
4. **轨迹可审计**：每次调用的 `ToolStep`（工具名/参数/成功与否/结果摘要）都随响应返回并落历史，
   出了问题可以复盘模型每一步做了什么。

---

## 六个工具

| 工具 | 作用 |
|---|---|
| `search_knowledge` | 在产品资料、MSDS、法规文件中检索，回答成分、限量、毒理、法规问题 |
| `query_order` | 查询订单状态、物流、运单号 |
| `search_formulation_ingredients` | 按功效/功能查找候选原料，返回用量区间、相态、配伍注意 |
| `get_formulation_ingredient` | 查单个原料的用量区间、工艺、配伍禁忌 |
| `build_formula` | 把选定原料组装成配方草案（自动补齐、补足 100%、合规校验） |
| `finish` | 显式结束一次推理，提交最终回复与意图 |

工具层不含业务逻辑，只做三件事：声明 JSON Schema 告诉模型工具长什么样；把模型参数校验后转给现有服务；
把执行结果裁剪成简短文本回填（原始命中块很长，直接塞回会吃满上下文）。结果文本有截断上限，
既够模型判断，又不会撑爆上下文。

---

## 数据资产（可开源部分）

知识库原始 PDF / PPTX 因体积与含未上市配方资料，**不随仓库分发**（已在 `.gitignore` 排除）。
开源可复现的核心是 5 份结构化配置：

| 文件 | 内容 |
|---|---|
| `data/formulation/ingredient_library.yaml` | 65 种原料：功能分类、功效标签、用量区间、相态、工艺、配伍禁忌、证据来源 |
| `data/formulation/product_templates.yaml` | 8 类剂型骨架（精华/面霜/乳液/眼霜/眼膜/面膜/洁面/化妆水） |
| `data/knowledge/glossary/ingredients.yaml` | 中英成分术语表，双向打通中文提问与英文 MSDS |
| `data/compliance/banned_claims.yaml` | 合规词库：block / rewrite / info 三级 |
| `data/knowledge/02_technology/deck_labels.yaml` | 英文技术页的中文主题映射 |

要跑通完整 RAG 检索，需自行放入对应的原始文档（README 早期版本曾记录资料来源构成：
《化妆品安全技术规范》563 页、BIOCELLINE 电子皮肤资料 42 页、14 份原料 MSDS，共 1701 个知识块）。
抽取式回答与配方研制在仅有结构化配置时即可演示。

---

## 关键设计决策（面试可展开）

### 1. 中英跨语言检索靠术语表，不靠模型碰运气
一半资料是英文 MSDS，提问是中文。术语表双向打通：入库时把中文成分名写进每个切块表头
（`透明质酸钠（Sodium Hyaluronate）`）让中文能字面命中；检索时把中文属性词扩展成英文
（`分子量` → `Molecular Weight`）。加原料不用改代码。

### 2. 向量化文本 ≠ 检索文本
切块里带 `【文件标题｜中文名｜章节】` 词头方便人看，但**向量化只看章节级标签 + 正文**。
早期把文件标题也嵌入，导致同一文件所有块向量几乎相同，排序退化成随机。关键词侧同理，只保留章节名。

### 3. 三类资料三种切法
法规 800 字/块（表格密集，切散会丢"成分+限量+条件"）；MSDS 按标准十六节（一节即一个完整答案单元）；
技术 PPT 450 字/页取版式最上方文本作标题。

### 4. 护栏是主链路的一部分，不是事后补丁
三级词库：`block`（治疗/消炎/痤疮）整段拒答转人工；`rewrite`（最好/第一/纯天然）删命中句；
`info`（防晒/美白）留痕备查。敏感人群独立逻辑：不拒答但必须附就医或局部测试提示。
另有领域边界判定，超范围低置信度直接转人工，避免"公司营收多少"被法规条文糊弄。

### 5. Agent 路径与 RAG 路径共用后处理
护栏、免责声明、医疗话术、引用构造集中在 `ChatService`。Agent 开关只改变"怎么得到草稿"，
不改变"草稿怎么被审查"，因此灰度切换 Agent 时合规行为完全一致。

---

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/chat` | 问答主接口（含 Agent 轨迹与配方草案） |
| GET | `/api/v1/history` | 当前访客的会话列表 |
| GET | `/api/v1/history/{session_id}` | 单个会话详情 |
| DELETE | `/api/v1/history/{session_id}` | 删除会话 |
| GET | `/health` / `/ready` | 存活 / 就绪探针 |
| GET | `/api/v1/kb/status` | 知识库概览 |
| GET | `/console/` | 侧栏界面 |

请求需带 `X-Visitor-Id` 头做访客隔离。`ChatResponse` 关键字段：

```json
{
  "answer": "…",
  "intent": "formulation",
  "citations": [{"source": "…", "section": "…", "snippet": "…", "score": 0.84}],
  "confidence": 0.56,
  "needs_human": false,
  "guard_flags": [],
  "trajectory": [
    {"tool": "search_formulation_ingredients", "arguments": {"benefit": "抗皱"}, "ok": true, "result_summary": "…"},
    {"tool": "build_formula", "arguments": {"product_type": "eye_cream"}, "ok": true, "result_summary": "…"}
  ],
  "formula": {"product_label": "眼霜", "items": [{"name": "…", "pct": 2.0}], "total": 100.0},
  "trace_id": "8837eac5663e48df"
}
```

---

## 测试与评测

```bash
python -m evals.run_eval     # 20 条评测用例：msds / regulation / technology / redline / unknown
pytest tests/ -q             # 27 个单测：Agent / 配方引擎 / 路由 / 离线回归 / 历史
```

评测断言三层逐层收窄：`expect_sources`（检索到正确文件）→ `expect_sections`（命中正确章节）→
`expect_keywords`（事实确实出现在答案里）。**红线类任何一条失败即不可上线**（不看总通过率），
其余看总通过率（门槛 75%）。另有输出卫生断言：不能泄漏 `Mock` 占位文案、
不能泄漏 `【文件｜章节】` 表头、不能超过 800 字、数值型问题必须给出数字。退出码非 0 可直接挂 CI。

---

## 目录结构

```
cosmetic-agent/
├── app/
│   ├── api/            # 控制器：chat / health 路由
│   ├── core/           # 配置校验、错误体系、结构化日志
│   ├── ingestion/      # 解析 / 切块 / 术语表注入 / 入库流水线
│   ├── repositories/   # 知识库、订单仓储
│   ├── schemas/        # Pydantic 模型（ChatResponse / FormulaDraft / ToolStep）
│   └── services/
│       ├── agent_service.py       # ReAct 循环、步数/超时上限、轨迹收集
│       ├── toolkit.py             # 六个工具的 Schema 与执行
│       ├── formulation_service.py # 配方引擎：选料/补齐/闭合100%/配伍校验
│       ├── chat_service.py        # 分流 + 统一后处理（护栏/引用/免责）
│       ├── router.py              # 研制 / 查属性 / 查成品配方表 路由
│       ├── guardrail_service.py   # 合规护栏
│       ├── retrieval_service.py   # 混合检索 + RRF
│       └── history_store.py       # 按访客隔离的历史存储
├── console/            # 侧栏交互界面（工具轨迹 + 配方卡片 + 历史抽屉）
├── data/               # 结构化配置（原始文档需自行放入，见上）
├── evals/              # 评测用例
├── scripts/            # 实测脚本
└── tests/              # 单元测试
```

---

## 安全与隐私

- 真实密钥只在 `.env`（已加入 `.gitignore`），仓库仅含 `.env.example` 占位值；
- 访客对话历史写入 `artifacts/visitors/<vid>/`（已加入 `.gitignore`），不入库；
- 日志对 `api_key / authorization / password / token / phone / id_card` 等字段脱敏；
- 原始法规/手册/MSDS 与未上市配方资料不随仓库分发。

---

## 已知限制与后续

- **无鉴权与限流**：当前接口对内开放前需补（演示项目可接受）。
- **法规超长表格**：表 1（1388 项禁用组分）、表 4（准用防腐剂）PDF 抽取后列会打散，
  问具体成分限量容易只召回相邻行，建议抽成结构化表精确查询而非继续堆 RAG。
- **SSE 流式输出**：法规类检索块多、生成慢，坐席等待超 3 秒会弃用，后续可加流式。
- **话术质量**：20/20 评测证明的是检索/抽取/护栏逻辑正确，不证明生成话术质量；
  接真实模型后需重跑并补充真实工单问题集。

> 免责声明：本项目配方草案由 AI 依据公开原料常规用法生成，用量为行业经验参考区间而非法规限值，
> 不构成可直接投产的生产配方，须经研发与法规人员复核后方可使用。
