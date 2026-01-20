Here is a cleaned-up, polished, and properly structured version of your README section. It incorporates the compatibility notes, fixes minor formatting issues, removes redundancy, improves clarity, and integrates the LangChain version pinning in a logical place.

```markdown
## Setup & Running the Document Processing Pipeline

This project combines **PaddleOCR** (including PP-OCRv5 and advanced document understanding pipelines) with **LangChain** to perform OCR-based text extraction and subsequent processing (e.g., RAG, structured extraction, agent-based querying).

**Important compatibility note (January 2026)**  
PaddlePaddle 3.3.x contains a known CPU inference bug related to oneDNN / PIR attribute conversion in some advanced pipelines (e.g., PP-Structure, PaddleX).  
To avoid this issue, we **pin PaddlePaddle to version 3.2.0**, which is fully compatible with the latest PaddleOCR (3.3.x as of Jan 2026).

### Prerequisites
- Operating System: Linux (tested on Ubuntu), macOS, or Windows
- Git
- Internet connection (required for initial model downloads — ~500 MB–1 GB)
- Optional: NVIDIA GPU + compatible CUDA drivers (for 5–10× faster inference)

### Recommended Installation: Miniconda (isolates dependencies cleanly)

1. **Install Miniconda** (skip if already installed)
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
   bash miniconda.sh
   ```
   - Follow the prompts: accept the license, confirm installation path, and allow `conda init`.
   - Restart your terminal (or run `source ~/.bashrc`).
   - If prompted about Anaconda ToS, accept the channels:
     ```bash
     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
     ```

2. **Create and activate a dedicated environment**
   ```bash
   conda create -n paddle-ocr-env python=3.10 -y
   conda activate paddle-ocr-env
   ```

3. **Install PaddlePaddle (pinned version)**
   - **CPU (recommended for stability / first-time setup)**
     ```bash
     pip install paddlepaddle==3.2.0
     ```
   - **GPU** (if you have an NVIDIA GPU and CUDA installed)
     - Check your CUDA version first:
       ```bash
       nvidia-smi
       ```
     - Example for CUDA 12.6 (common in 2026):
       ```bash
       pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
       ```
     - For other CUDA versions, visit: https://www.paddlepaddle.org.cn/install/quick

4. **Install PaddleOCR**
   ```bash
   # Basic installation (PP-OCRv5 text detection + recognition)
   pip install paddleocr

   # Recommended: full features (tables, layout analysis, formulas, document understanding, PaddleOCR-VL)
   pip install "paddleocr[all]"
   ```
   → The first run of PaddleOCR will automatically download models.

5. **Install LangChain & project dependencies**
   To ensure compatibility with legacy agents (`create_tool_calling_agent`, `AgentExecutor`) and avoid version conflicts:

   ```bash
   pip install langchain==0.3.27 \
               langchain-core==0.3.72 \
               langchain-text-splitters==0.3.9 \
               langchain-classic==1.0.1 \
               langchain-anthropic
   ```

   Then install any additional project-specific packages:
   ```bash
   pip install -r requirements.txt
   ```
