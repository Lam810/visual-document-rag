# AscendFlow AI - 智能文档分析系统

一个基于人工智能的PDF文档分析系统,支持文档上传、解析、查询等功能。

## 功能特点

- PDF文档上传和解析
- Markdown格式预览
- 智能文档查询
- 实时处理进度显示
- 文档管理和浏览

## 技术栈

- 后端:Python + Flask
- 前端:HTML + CSS + JavaScript
- PDF处理:PDF-Extract-Kit
- 文档检索:RAG (Retrieval-Augmented Generation)
- UI框架:Bootstrap 5
- 图标:Font Awesome

## 安装说明

1. 克隆仓库
```bash
git clone https://github.com/Lam810/ascendflow-ai.git
cd ascendflow-ai
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
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

## 开发者

- Lam Tzar Tung

## 许可证

MIT License

## 致谢

感谢以下开源项目的支持:
- PDF-Extract-Kit
- Flask
- Bootstrap
- Font Awesome
