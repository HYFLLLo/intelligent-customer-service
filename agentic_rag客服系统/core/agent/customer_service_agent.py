import os
import uuid
from typing import List, Dict, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from rich.console import Console

console = Console()

class CustomerServiceAgent:
    def __init__(self, hybrid_retriever):
        """初始化客服Agent"""
        self.hybrid_retriever = hybrid_retriever
        self.llm = OllamaLLM(
            model=os.getenv("LLM_MODEL", "qwen3:8b"),
            temperature=0.3,
            top_p=0.9,
            base_url="http://localhost:11434"
        )
        self.output_parser = StrOutputParser()
        
        # 初始化提示模板
        self.plan_prompt = ChatPromptTemplate.from_template("""
你是一个专业的客服系统规划师，需要分析用户问题并制定解决方案。

用户问题：{query}

请制定一个详细的解决方案，包括：
1. 问题分析：用户的核心需求是什么？
2. 所需信息：需要从知识库中检索哪些信息？
3. 工具使用：需要调用哪些工具？调用顺序是什么？
4. 执行计划：如何一步步解决用户问题？

请以清晰的步骤列出你的规划。
""")
        
        self.react_prompt = ChatPromptTemplate.from_template("""
你是一个专业的客服助手，需要根据规划执行任务并解决用户问题。

用户问题：{query}

执行计划：
{plan}

已检索的信息：
{retrieved_info}

请按照以下格式执行任务：
1. 思考：分析当前状态和下一步行动
2. 行动：执行具体操作（如检索信息）
3. 反思：评估行动结果，调整后续步骤

最终，你需要：
- 基于检索到的信息提供准确回答
- 结构化呈现结果（分点+加粗关键步骤）
- 标注信息来源（如"[知识库文档：XXX]"）
- 确保回答专业、友好、有条理
""")
        
        self.summarize_prompt = ChatPromptTemplate.from_template("""
请将以下对话和信息总结为一个结构化的回答：

用户问题：{query}

Agent执行过程：
{execution_process}

检索到的相关信息：
{retrieved_info}

最终回答需要：
1. 直接回应用户问题
2. 分点呈现关键信息
3. 加粗重要步骤
4. 标注信息来源
5. 语言友好、专业
6. 基于检索到的信息提供准确回答，不要编造信息
""")
    
    def process_query(self, query: str) -> Dict:
        """处理用户查询"""
        try:
            query_id = str(uuid.uuid4())
            logger.info(f"处理用户查询: {query_id}, 内容: {query}")
            
            # 1. 制定执行计划
            plan = self._create_plan(query)
            logger.info(f"生成执行计划: {plan}")
            
            # 2. 执行ReAct推理循环
            execution_process, retrieved_info = self._execute_react(query, plan)
            logger.info(f"执行完成，检索到 {len(retrieved_info)} 条信息")
            
            # 3. 生成最终回答
            final_answer = self._generate_final_answer(query, execution_process, retrieved_info)
            
            # 4. 构建返回结果
            result = {
                "query_id": query_id,
                "query": query,
                "answer": final_answer,
                "execution_process": execution_process,
                "retrieved_info": retrieved_info,
                "timestamp": os.path.getmtime(__file__)
            }
            
            return result
        except Exception as e:
            logger.error(f"处理查询失败: {str(e)}")
            return {
                "query_id": str(uuid.uuid4()),
                "query": query,
                "answer": f"抱歉，处理您的问题时出现错误：{str(e)}",
                "execution_process": [],
                "retrieved_info": [],
                "timestamp": os.path.getmtime(__file__)
            }
    
    def _create_plan(self, query: str) -> str:
        """制定执行计划"""
        try:
            plan_chain = self.plan_prompt | self.llm | self.output_parser
            plan = plan_chain.invoke({"query": query})
            return plan
        except Exception as e:
            logger.error(f"制定计划失败: {str(e)}")
            return "默认计划：检索相关信息并生成回答"
    
    def _execute_react(self, query: str, plan: str) -> tuple:
        """执行ReAct推理循环"""
        execution_process = []
        retrieved_info = []
        
        try:
            # 步骤1：思考
            thought = "分析用户问题，确定需要检索的关键信息"
            execution_process.append({"step": "思考", "content": thought})
            
            # 步骤2：行动 - 检索信息
            action = "调用混合检索工具获取相关信息"
            execution_process.append({"step": "行动", "content": action})
            
            # 执行检索
            retrieved_docs = self.hybrid_retriever.get_relevant_documents(query)
            
            # 格式化检索结果
            for i, doc in enumerate(retrieved_docs):
                doc_info = {
                    "id": i + 1,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": doc.get("score", 0)
                }
                retrieved_info.append(doc_info)
            
            # 步骤3：反思
            reflection = f"检索完成，找到 {len(retrieved_info)} 条相关信息，现在基于这些信息生成回答"
            execution_process.append({"step": "反思", "content": reflection})
            
            return execution_process, retrieved_info
        except Exception as e:
            logger.error(f"执行ReAct失败: {str(e)}")
            execution_process.append({"step": "错误", "content": str(e)})
            return execution_process, retrieved_info
    
    def _generate_final_answer(self, query: str, execution_process: List[Dict], retrieved_info: List[Dict]) -> str:
        """生成最终回答"""
        try:
            # 构建执行过程字符串
            process_str = "\n".join([f"{step['step']}: {step['content']}" for step in execution_process])
            
            # 格式化检索信息
            retrieved_info_str = self._format_retrieved_info(retrieved_info)
            
            # 生成回答
            summarize_chain = self.summarize_prompt | self.llm | self.output_parser
            answer = summarize_chain.invoke({
                "query": query,
                "execution_process": process_str,
                "retrieved_info": retrieved_info_str
            })
            
            return answer
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return f"抱歉，生成回答时出现错误：{str(e)}"
    
    def _format_retrieved_info(self, retrieved_info: List[Dict]) -> str:
        """格式化检索信息"""
        if not retrieved_info:
            return "无相关信息"
        
        formatted_info = []
        for i, info in enumerate(retrieved_info):
            source = info["metadata"].get("source", "未知来源")
            content = info["content"][:200] + "..." if len(info["content"]) > 200 else info["content"]
            formatted_info.append(f"[{i+1}] 来源: {source}\n内容: {content}")
        
        return "\n\n".join(formatted_info)
    
    def _validate_query(self, query: str) -> bool:
        """验证查询是否合法"""
        # 简单的安全检查
        forbidden_topics = ["政治", "隐私", "违法", "恶意"]
        for topic in forbidden_topics:
            if topic in query:
                return False
        return True
    
    def _handle_unsafe_query(self) -> Dict:
        """处理不安全查询"""
        return {
            "query_id": str(uuid.uuid4()),
            "answer": "抱歉，我无法处理此类问题。请提供与客服相关的合法合规问题。",
            "execution_process": [],
            "retrieved_info": [],
            "timestamp": os.path.getmtime(__file__)
        }
