from flask import Flask, request, render_template, jsonify
import os
import shutil
import subprocess
from rag import RAGSystem

app = Flask(__name__)
rag_system = RAGSystem()

# 确保必要的目录存在
os.makedirs('input', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('data', exist_ok=True)

@app.route('/')
def index():
    # 获取data目录下的所有markdown文件
    md_files = []
    for filename in os.listdir('data'):
        if filename.endswith('.md'):
            md_files.append(filename)
    return render_template('index.html', documents=md_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    # 计算预估时间(基于文件大小)
    file_size = len(file.read())
    file.seek(0)  # 重置文件指针
    # 假设每MB需要30秒处理时间
    estimated_time = (file_size / (1024 * 1024)) * 30  

    # 保存上传的文件
    input_path = os.path.join('input', file.filename)
    file.save(input_path)
    
    try:
        # 运行PDF处理脚本
        subprocess.run([
            'python', 
            './PDF-Extract-Kit/project/pdf2markdown/scripts/run_project.py',
            '--config', 
            './pdf2markdown.yaml'
        ], check=True)
        
        # 移动生成的markdown文件到data目录
        output_md = os.path.join('output', os.path.splitext(file.filename)[0] + '.md')
        if os.path.exists(output_md):
            shutil.move(output_md, os.path.join('data', os.path.basename(output_md)))
            
            # 重新初始化RAG系统以包含新文件
            global rag_system
            rag_system = RAGSystem()
            rag_system.process_documents()
            
            # 获取更新后的文件列表和新文档的markdown内容
            md_files = [f for f in os.listdir('data') if f.endswith('.md')]
            
            # 读取新生成的markdown文件内容
            new_md_path = os.path.join('data', os.path.splitext(file.filename)[0] + '.md')
            markdown_content = ''
            if os.path.exists(new_md_path):
                with open(new_md_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
            
            return jsonify({
                'message': 'File processed successfully',
                'documents': md_files,
                'estimated_time': estimated_time,
                'markdown_content': markdown_content
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    try:
        md_files = [f for f in os.listdir('data') if f.endswith('.md')]
        return jsonify({'documents': md_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        results = rag_system.search(data['question'])
        return jsonify({
            'results': [
                {'content': doc.page_content}
                for doc in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/markdown/<filename>')
def get_markdown(filename):
    try:
        file_path = os.path.join('data', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({'content': content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 初始化RAG系统
    rag_system.process_documents()
    # 在6006端口运行
    app.run(host='0.0.0.0', port=6006, debug=True)
