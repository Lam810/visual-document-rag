# AscendFlow AI - 智能文档分析系统

一个基于人工智能的PDF文档分析系统,支持文档上传、解析、查询等功能。通过先进的PDF解析技术和RAG检索增强生成技术,实现文档的智能化管理和查询。

## 背景

- 视觉丰富文档（VRDs）结合了复杂信息，将文本与图形、图表和表格等视觉元素融合在一起，现有大模型很难完全读取。
- 与传统的文本文档不同，VRDs有两个主要特征：与排版细节（例如字体、大小、样式、颜色）相关联的文本，以空间方式组织信息的布局，以及增强理解所需的视觉元素，如图表和图形。
- 因此，我提供了这个项目，使用多个大模型协同，先对复杂的pdf文档进行布局检测，然后采用不同的工具，识别公示、表格和图片，并转化成Latex或者markdown格式，交给向量模型变成RAG系统，并最终实现大模型进行理解和学习。

## 功能特点

- **PDF文档处理**
  - 支持PDF文件上传和自动解析
  - 实时显示处理进度
  - 自动转换为Markdown格式

- **文档预览**
  - 实时Markdown预览
  - 支持文档目录导航
  - 优雅的阅读体验

- **智能检索**
  - 基于RAG的智能问答
  - 精准定位相关内容
  - 自然语言交互

- **文档管理**
  - 已处理文档列表
  - 快速切换不同文档
  - 文档永久保存

## 技术栈

- **后端**
  - Python + Flask
  - RAG (Retrieval-Augmented Generation)
  - PDF-Extract-Kit

- **前端**
  - HTML + CSS + JavaScript
  - Bootstrap 5
  - Font Awesome

## 安装说明

1. 克隆项目仓库
```bash
git clone https://github.com/Lam810/visual-document-rag.git
cd visual-document-rag
```

2. 安装PDF-Extract-Kit
```bash
# 克隆PDF-Extract-Kit仓库
git clone https://github.com/opendatalab/PDF-Extract-Kit.git

# 安装Git LFS并下载模型
git lfs install
git clone https://www.modelscope.cn/opendatalab/pdf-extract-kit-1.0.git

# 移动模型文件
mv ./pdf-extract-kit-1.0/models ./PDF-Extract-Kit
```

3. 配置Python环境
```bash
# 创建并激活conda环境
conda create -n pdf-extract-kit-1.0 python=3.10
conda activate pdf-extract-kit-1.0

# 安装项目依赖
pip install -r requirements.txt
```

4. 运行应用
```bash
python app.py
```

应用将在 http://localhost:6006 运行

## 目录结构

```
.
├── app.py              # Flask应用主文件
├── rag.py             # RAG系统实现
├── requirements.txt    # 项目依赖
├── templates/         # HTML模板
│   └── index.html    # 主页面
├── data/             # 处理后的markdown文件存储目录
├── input/            # 上传的PDF文件存储目录
└── output/           # 临时输出文件目录
```

## 使用说明

1. 上传PDF文件
   - 点击"上传PDF文件"按钮选择文件
   - 系统会显示预估处理时间
   - 等待处理完成

2. 查看文档
   - 在左侧文档列表中选择要查看的文档
   - 右侧会显示文档的Markdown内容
   - 可以滚动浏览全文

3. 智能查询
   - 在查询框输入问题
   - 系统会基于文档内容进行回答
   - 支持自然语言提问

## 开发者

- Lam Tzar Tung

## 许可证

MIT License

## 致谢

感谢以下开源项目的支持:
- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)
- [Flask](https://flask.palletsprojects.com/)
- [Bootstrap](https://getbootstrap.com/)
- [Font Awesome](https://fontawesome.com/)
