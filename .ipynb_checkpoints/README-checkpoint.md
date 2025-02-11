# PDF文档RAG系统

这是一个基于RAG(检索增强生成)的PDF文档处理和问答系统。系统可以将PDF文档转换为markdown格式,并支持基于文档内容的智能问答。

## 功能特点

- PDF文档转Markdown:自动将上传的PDF文件转换为markdown格式
- 文档向量化:使用先进的向量嵌入模型处理文档内容
- 智能问答:基于文档内容的相似度检索和问答功能
- 实时进度显示:文件处理进度条和预估时间显示
- 文档管理:已处理文档列表展示

## 系统要求

- Python 3.8+
- 虚拟环境(推荐使用conda或venv)

## 安装步骤

1. 克隆项目并创建虚拟环境:
```bash
git clone [项目地址]
cd [项目目录]
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 运行应用:
```bash
python app.py
```

应用将在 http://localhost:6006 启动

## 目录结构

```
.
├── app.py              # Flask应用主文件
├── rag.py             # RAG系统实现
├── requirements.txt    # 项目依赖
├── templates/         # HTML模板
│   └── index.html    # 主页面模板
├── data/             # 处理后的markdown文件存储目录
├── input/            # PDF文件上传目录
└── output/           # 临时输出目录
```

## 使用说明

1. 启动应用后,访问 http://localhost:6006
2. 在"上传PDF文件"区域选择要处理的PDF文件
3. 系统会显示预估处理时间,并开始处理文件
4. 处理完成后,文件会出现在"已处理的文档"列表中
5. 在查询框中输入问题,系统会基于文档内容进行回答

## 注意事项

1. 文件处理时间与PDF文件大小和复杂度相关
2. 建议每个PDF文件大小不超过50MB
3. 目前仅支持中文PDF文档
4. 需要确保系统有足够的存储空间用于处理文件
5. 首次运行时会下载必要的模型文件,需要保持网络连接

## 常见问题

1. 如果遇到模型下载失败,请检查网络连接并重试
2. 如果文件处理失败,请检查PDF文件是否损坏或格式是否支持
3. 如果查询没有返回结果,可能是问题与文档内容相关度不够

## 技术栈

- Flask: Web框架
- LangChain: RAG实现
- ChromaDB: 向量数据库
- ModelScope: 文本向量化模型
- PyMuPDF: PDF处理
