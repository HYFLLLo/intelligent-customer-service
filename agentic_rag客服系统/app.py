from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
import json
from dotenv import load_dotenv

# 加载配置
load_dotenv()

# 导入核心模块
from core.knowledge_base import KnowledgeBaseManager
from core.retrieval import HybridRetriever
from core.agent import CustomerServiceAgent
from core.feedback import FeedbackManager

# 初始化应用
app = FastAPI(
    title="Agentic RAG 本地智能客服系统",
    description="基于Ollama本地模型的智能客服系统，支持本地知识库管理、混合检索和透明化工具调用",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应设置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化核心组件
knowledge_base = KnowledgeBaseManager()
hybrid_retriever = HybridRetriever(knowledge_base)
agent = CustomerServiceAgent(hybrid_retriever)
feedback_manager = FeedbackManager()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="./web"), name="static")

# 根路径返回前端页面
@app.get("/")
async def root():
    return FileResponse("./web/index.html")

# 健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 上传文档
@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_type: str = Form(...)
):
    try:
        file_path = os.path.join(os.getenv("DOCUMENTS_PATH", "./data/documents"), file.filename)
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 处理文档并添加到知识库
        knowledge_base.add_document(file_path, document_type)
        
        return {"message": "文档上传成功", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 处理用户查询
@app.post("/api/query")
async def process_query(query: dict):
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="查询内容不能为空")
        
        # 使用Agent处理查询
        result = agent.process_query(user_query)
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 提交反馈
@app.post("/api/feedback")
async def submit_feedback(feedback: dict):
    try:
        query_id = feedback.get("query_id")
        is_solved = feedback.get("is_solved")
        additional_info = feedback.get("additional_info", "")
        
        # 处理反馈
        feedback_manager.add_feedback(query_id, is_solved, additional_info)
        
        return {"message": "反馈提交成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取反馈统计
@app.get("/api/feedback/stats")
async def get_feedback_stats():
    try:
        stats = feedback_manager.get_feedback_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取知识库统计
@app.get("/api/documents/stats")
async def get_documents_stats():
    try:
        # 获取文档目录中的文件列表
        documents_path = os.getenv("DOCUMENTS_PATH", "./data/documents")
        files = os.listdir(documents_path)
        # 过滤出实际的文档文件（排除临时文件和隐藏文件）
        document_files = [f for f in files if not f.startswith('.') and os.path.isfile(os.path.join(documents_path, f))]
        return {
            "total_documents": len(document_files),
            "documents": document_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取知识库内容
@app.get("/api/knowledge-base")
async def get_knowledge_base():
    try:
        # 获取文档目录中的文件列表
        documents_path = os.getenv("DOCUMENTS_PATH", "./data/documents")
        files = os.listdir(documents_path)
        # 过滤出实际的文档文件（排除临时文件和隐藏文件）
        document_files = [f for f in files if not f.startswith('.') and os.path.isfile(os.path.join(documents_path, f))]
        
        # 构建知识库内容
        knowledge_base_content = {
            "documents": document_files,
            "total_documents": len(document_files)
        }
        
        return knowledge_base_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动应用
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
