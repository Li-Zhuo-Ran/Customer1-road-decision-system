# 路面养护决策系统

> 基于 Streamlit 的路面病害评估、PCI/PSSI 年度预测与专业化养护方案生成工具。

---

## 一句话介绍

本项目将测量表（平整度/构造深度 + 车辙）转换为 PCI/PSSI 基线与多年份预测，结合区域专家知识库自动生成养护决策与可下载的专业 Word 技术方案，适合养护单位快速出具技术交底与方案建议。

## 亮点

- 支持两表合并（按序号/桩号）并计算 PCI/PSSI 基线（启发式），也支持 PSO 优化的 XGBoost 回归器进行有标签训练。
- 按年生成 PCI/PSSI 预测并可视化，方便做长期维护规划。
- 结合专家知识库生成决策方案，直接导出 Word 报告（含核心参数与施工要点）。

---

## 快速检查（本地测试前准备）

1. 推荐 Python 版本：Python 3.10 - 3.12。
2. 在项目根运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. 运行并在浏览器打开：

```bash
streamlit run app.py
# 打开 http://localhost:8501
```

4. 在 `M1` 上传或生成示例测量表/车辙表，逐步测试 M2 → M3 → M4 流程。

---

## 部署：方案 C — 部署到 Streamlit Community Cloud / 企业版（详细步骤）

适用场景：需要快速公开 demo、或把应用托管给非技术同事通过浏览器直接访问。推荐用于演示与小规模内部使用。

前置条件：将代码推送到 GitHub（私有或公有仓库），并包含 `requirements.txt` 与 `app.py`。

步骤如下：

1) 准备仓库

- 在本地确认项目可运行（见“快速检查”）。
- 把代码推到 GitHub：

```bash
git init
git add .
git commit -m "Initial commit: RoadSystem"
git remote add origin <your-repo-url>
git push -u origin main
```

2) 配置 `requirements.txt`

- 确保 `requirements.txt` 中列出运行时依赖（例如 `streamlit`, `pandas`, `xgboost` 等）。我已为你添加了推荐的依赖与最低版本。
- 如果部分依赖体积较大（例如 XGBoost），可以在部署前测试是否能在 Streamlit Cloud 的构建时间与内存限制内安装成功；否则考虑去除模型训练逻辑、仅保留推理或托管模型到远端 API。

3) 在 Streamlit Community Cloud 创建应用

- 登录到 https://share.streamlit.io 并关联你的 GitHub 账户（Streamlit for Teams / Enterprise 则由管理员配置）。
- 点击 “New app” → 选择仓库、分支（通常 `main`）和应用入口（填写 `app.py`）。
- 点击 Deploy，平台会自动执行构建（安装 `requirements.txt`）。构建过程会在界面显示日志，若构建失败请根据日志调整依赖或分拆功能。

4) 环境变量与私密信息

- 如果需要保存私密配置（例如外部模型 URL、API key、数据库凭据），在 Streamlit Cloud 的应用设置中配置 Secrets：

```text
# 在 share.streamlit.io 的 App → Settings → Secrets 中添加
MY_API_KEY="xxxx"
```

- 在代码中通过 `st.secrets['MY_API_KEY']` 获取。

5) 构建失败常见问题与解决方案

- 构建超时或内存错误：减少要安装的包，考虑把训练部分移出（改为在本地或服务器训练并把模型文件放在远端），或使用轻量模型。
- 二进制依赖编译失败（如 XGBoost）：尝试指定已有的 wheel 版本或使用纯 Python 替代（例如用轻量 sklearn 模型）作为过渡。
- 大文件超限：把大模型文件、示例数据从仓库移到云存储（S3）或 Git LFS，再在运行时下载。

6) 发布与访问

- 部署成功后会得到一个公开访问的 URL（或仅限团队访问的 URL）。把该链接分享给使用者，用户即可通过浏览器访问应用，无需安装任何软件。

---

## 建议的生产实践

- 把模型训练（耗时/资源高）与线上推理分离：在 CI 或独立计算环境训练模型并把训练好的模型文件托管（云存储或模型服务），Streamlit 只做推理。
- 添加监控与日志（记录用户上传的数据样本、模型预测统计），便于后期回溯与模型迭代。
- 提供 `examples/` 目录放示例测量表（`meas_example.csv`、`rut_example.csv`）与一份 `data_dictionary.md` 描述每列含义与格式。

---

## 我能继续帮你的事

- 我可以生成并提交：
  - `run_app.py`（本地启动脚本）和 `dev-requirements.txt`（包含 `pyinstaller` 等打包依赖）；或
  - 在你授权下，我可以尝试本地用 `PyInstaller` 打包并调试（需要在你的机器执行）；或
  - 直接在仓库中创建 `examples/` 数据样例与 `data_dictionary.md`。

请选择你希望我接着做的项（例如“生成 run_app.py 与 dev-requirements.txt”）。

---

## 主要功能

- 多源测量数据接入：支持 `平整度与构造深度` 与 `车辙` 两表按序号/桩号对齐并合并。
- 启发式与机器学习预测：提供基线启发式计算与可选的 PSO 优化 + XGBoost 回归器分别预测 `PCI` 与 `PSSI`。
- 按年预测：生成多年份（可配置）PCI/PSSI 预测并可视化。
- 专家决策规则：基于区域专家知识库映射到 `雾封层` / `同步碎石封层` / `局部铣刨重铺` / `结构大修` 等对策，并输出核心参数/施工窗口/预期效果。
- 专业 Word 导出：生成包含技术参数、施工关键控制点的可下载 Word 报告（`docx`）。

## 适用场景

- 交通管理/养护单位进行路面性能批量评估与快速方案生成。
- 需要将测量数据快捷转化为可执行养护方案与技术交底文档的场景。

## 目录（主要文件）

- `app.py`：主应用（Streamlit）。
- `data.csv`：示例/模板数据（若存在）。
- `requirements.txt`：依赖列表（用于虚拟环境与打包参考）。

## 快速开始（开发/测试）

1. 克隆或下载项目到本地：

```bash
git clone <repo-url>
cd RoadSystem
```

2. 建议使用虚拟环境并安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. 运行 Streamlit 应用：

```bash
streamlit run app.py
```

4. 打开浏览器访问 http://localhost:8501

---

## 将应用部署为“无需用户安装 Python”的可下载软件（概述与推荐方法）

说明：Streamlit 是个基于 Python 的 Web 应用框架。要把应用打包为“用户无需安装 Python 就能双击运行”的软件，常见方法是把带有 Python 运行时的程序打包成独立可执行文件或应用程序包。下面给出几种常用方案及操作步骤与注意事项。

### 方案 A — 使用 PyInstaller 打包成单文件/应用（适用于 macOS/Windows/Linux）

优点：相对简单，可生成单个可执行文件或 App 包；用户无需安装 Python。
缺点：依赖项复杂时（如 XGBoost、Plotly、Streamlit）可能需要额外参数和调试；不同平台需在对应平台上构建（在 macOS 上为 macOS 构建，在 Windows 上为 Windows 构建）。

步骤要点（macOS 示例）：

1. 在项目根创建一个启动脚本 `run_app.py`（示例见下），它会在内部启动 Streamlit：

```python
# run_app.py
import subprocess
import webbrowser
import time

port = 8501
subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', str(port)])
time.sleep(2)
webbrowser.open(f'http://localhost:{port}')
```

2. 安装 PyInstaller：

```bash
pip install pyinstaller
```

3. 使用 PyInstaller 打包（示例生成一个目录式可运行 app）：

```bash
pyinstaller --name RoadSystem --add-data "./:./" --hidden-import=sklearn --hidden-import=xgboost run_app.py
```

说明与调优：

- `--add-data` 用来包含静态资源和 `app.py` 等文件；路径格式在 Windows/macOS/Linux 不同，注意 PyInstaller 文档。
- 可能需要通过 `--hidden-import` 指定若干隐式依赖（例如 `sklearn`、`xgboost` 等）。
- 打包后会生成 `dist/RoadSystem` 目录，目录内为可执行文件与依赖；将该目录打包成 `.zip` 或 macOS `.app` 供用户下载。

注意事项：

- XGBoost 与某些二进制库在打包时容易出问题，需要逐一解决缺失的动态库；若遇到问题，建议先在本地脚本中确认 `run_app.py` 能在虚拟环境中正确启动再打包。

### 方案 B — 使用 Docker 打包并发布镜像（适合发布到企业内部服务器）

优点：环境一致性好，适合运维部署；用户无需本地运行 Python，只需运行容器。

示例 Dockerfile：

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port","8501","--server.headless","true"]
```

构建并运行：

```bash
docker build -t roadsystem:latest .
docker run -p 8501:8501 roadsystem:latest
```

用户下载与运行只需 Docker 环境，或你可以把镜像部署到私有 registry / 云平台，用户通过浏览器访问 URL 即可。

### 方案 C — 部署到 Streamlit Community Cloud / 企业版

优点：部署最简单，直接托管在 Streamlit 平台；适合演示和小规模使用。
缺点：需要将代码推送到 GitHub，并遵循平台资源限制。

步骤：将仓库推送到 GitHub，然后在 Streamlit Community Cloud 新建应用，选择仓库与主分支并填写 `streamlit run app.py`。平台会自动安装 `requirements.txt`。

---
