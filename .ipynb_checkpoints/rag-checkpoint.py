from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain_community.vectorstores import Chroma
import os

class RAGSystem:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        # 使用modelscope的文本向量模型,并指定缓存目录
        os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope'
        self.embeddings = ModelScopeEmbeddings(
            model_id="damo/nlp_corom_sentence-embedding_chinese-base"
        )
        self.text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
        self.db = None

    def load_markdown_files(self):
        """加载所有markdown文件"""
        documents = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.md'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
        return documents

    def process_documents(self):
        """处理文档:加载、分块、向量化存储"""
        print("Loading documents...")
        documents = self.load_markdown_files()
        
        print("Splitting text...")
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_text(doc))
        
        print(f"Created {len(chunks)} chunks")
        
        print("Creating vector store...")
        # 将向量数据库保存在autodl-tmp目录下
        self.db = Chroma.from_texts(
            chunks,
            self.embeddings,
            persist_directory="/root/autodl-tmp/chroma_db"
        )
        print("Vector store created successfully!")

    def search(self, query, k=3):
        """搜索相似内容"""
        if not self.db:
            raise ValueError("Please process documents first!")
        
        results = self.db.similarity_search(query, k=k)
        return results

def main():
    # 初始化RAG系统
    rag = RAGSystem()
    
    # 处理文档
    rag.process_documents()
    
    # 交互式搜索
    while True:
        query = input("\n请输入搜索问题 (输入'q'退出): ")
        if query.lower() == 'q':
            break
            
        results = rag.search(query)
        print("\n相关内容:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- 结果 {i} ---")
            print(doc.page_content)

if __name__ == "__main__":
    main()
