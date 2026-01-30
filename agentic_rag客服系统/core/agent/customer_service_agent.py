import os
import uuid
import asyncio
from typing import List, Dict, Optional, AsyncGenerator
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from rich.console import Console

from core.memory.conversation_memory import ConversationMemory

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
        self.conversation_memory = ConversationMemory()
        
        # 初始化提示模板
        self.plan_prompt = ChatPromptTemplate.from_template("""
你是一个专业的客服系统规划师，需要分析用户问题并制定解决方案。

{conversation_history}

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

{conversation_history}

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
- 考虑对话历史中的上下文信息
""")
        
        self.summarize_prompt = ChatPromptTemplate.from_template("""
请将以下对话和信息总结为一个结构化的回答：

{conversation_history}

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
7. 考虑对话历史中的上下文信息
""")
    
    def process_query(self, query: str, session_id: str = "default") -> Dict:
        """处理用户查询"""
        try:
            query_id = str(uuid.uuid4())
            logger.info(f"处理用户查询: {query_id}, 内容: {query}, 会话: {session_id}")
            
            # 获取对话历史和个人信息
            conversation_history = self.conversation_memory.get_context(session_id)
            personal_info = self.conversation_memory.get_personal_info(session_id)
            
            # 合并上下文
            full_context = conversation_history
            if personal_info:
                full_context = personal_info + "\n" + conversation_history
            
            # 1. 制定执行计划
            plan = self._create_plan(query, full_context)
            logger.info(f"生成执行计划: {plan}")
            
            # 2. 执行ReAct推理循环
            execution_process, retrieved_info = self._execute_react(query, plan, full_context)
            logger.info(f"执行完成，检索到 {len(retrieved_info)} 条信息")
            
            # 3. 生成最终回答
            final_answer = self._generate_final_answer(query, execution_process, retrieved_info, full_context)
            
            # 保存对话到记忆
            self.conversation_memory.add_message(session_id, "user", query)
            self.conversation_memory.add_message(session_id, "assistant", final_answer)
            
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
    
    async def process_query_stream(self, query: str, query_id: str, session_id: str = "default") -> AsyncGenerator[Dict, None]:
        """流式处理用户查询"""
        try:
            logger.info(f"流式处理用户查询: {query_id}, 内容: {query}, 会话: {session_id}")
            
            # 获取对话历史和个人信息
            conversation_history = self.conversation_memory.get_context(session_id)
            personal_info = self.conversation_memory.get_personal_info(session_id)
            
            # 合并上下文
            full_context = conversation_history
            if personal_info:
                full_context = personal_info + "\n" + conversation_history
            
            # 步骤1：制定执行计划
            yield {"type": "step", "step": "规划", "content": "正在分析您的问题并制定执行计划..."}
            await asyncio.sleep(0.5)
            
            # 流式生成执行计划
            yield {"type": "step", "step": "规划", "content": "1. 分析用户问题的核心需求..."}
            await asyncio.sleep(0.3)
            
            yield {"type": "step", "step": "规划", "content": "2. 确定需要检索的关键信息..."}
            await asyncio.sleep(0.3)
            
            yield {"type": "step", "step": "规划", "content": "3. 制定工具调用策略和顺序..."}
            await asyncio.sleep(0.3)
            
            # 实际生成计划
            plan = self._create_plan(query, full_context)
            logger.info(f"生成执行计划: {plan}")
            
            # 流式显示计划内容
            plan_lines = plan.split('\n')
            for i, line in enumerate(plan_lines):
                if line.strip():
                    yield {"type": "step", "step": "规划", "content": f"执行计划: {line.strip()}"}
                    await asyncio.sleep(0.2)
            
            yield {"type": "step", "step": "规划", "content": "执行计划已生成，开始执行..."}
            await asyncio.sleep(0.3)
            
            # 步骤2：思考
            yield {"type": "step", "step": "思考", "content": "分析用户问题，确定需要检索的关键信息"}
            await asyncio.sleep(0.5)
            
            # 步骤3：行动 - 检索信息
            yield {"type": "step", "step": "行动", "content": "调用混合检索工具获取相关信息"}
            await asyncio.sleep(0.3)
            
            # 执行检索
            retrieved_docs = self.hybrid_retriever.get_relevant_documents(query)
            
            # 格式化检索结果
            retrieved_info = []
            for i, doc in enumerate(retrieved_docs):
                doc_info = {
                    "id": i + 1,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": doc.get("score", 0)
                }
                retrieved_info.append(doc_info)
            
            # 步骤4：反思
            yield {"type": "step", "step": "反思", "content": f"检索完成，找到 {len(retrieved_info)} 条相关信息，现在基于这些信息生成回答"}
            await asyncio.sleep(0.5)
            
            # 步骤5：生成最终回答（流式）
            yield {"type": "step", "step": "生成回答", "content": "正在生成最终回答..."}
            
            # 构建执行过程字符串
            execution_process = [
                {"step": "思考", "content": "分析用户问题，确定需要检索的关键信息"},
                {"step": "行动", "content": "调用混合检索工具获取相关信息"},
                {"step": "反思", "content": f"检索完成，找到 {len(retrieved_info)} 条相关信息，现在基于这些信息生成回答"}
            ]
            
            process_str = "\n".join([f"{step['step']}: {step['content']}" for step in execution_process])
            
            # 格式化检索信息
            retrieved_info_str = self._format_retrieved_info(retrieved_info)
            
            # 流式生成回答
            full_answer = ""
            answer_chunks = await self._generate_final_answer_stream(query, process_str, retrieved_info_str, full_context)
            
            for chunk in answer_chunks:
                full_answer += chunk
                yield {"type": "answer_chunk", "content": chunk}
                await asyncio.sleep(0.05)
            
            # 保存对话到记忆
            self.conversation_memory.add_message(session_id, "user", query)
            self.conversation_memory.add_message(session_id, "assistant", full_answer)
            
            # 发送完成信号
            yield {
                "type": "complete",
                "data": {
                    "query_id": query_id,
                    "query": query,
                    "answer": full_answer,
                    "execution_process": execution_process,
                    "retrieved_info": retrieved_info,
                    "timestamp": os.path.getmtime(__file__)
                }
            }
            
        except Exception as e:
            logger.error(f"流式处理查询失败: {str(e)}")
            yield {"type": "error", "data": str(e)}
    
    def _create_plan(self, query: str, conversation_history: str = "") -> str:
        """制定执行计划"""
        try:
            plan_chain = self.plan_prompt | self.llm | self.output_parser
            plan = plan_chain.invoke({"query": query, "conversation_history": conversation_history})
            return plan
        except Exception as e:
            logger.error(f"制定计划失败: {str(e)}")
            return "默认计划：检索相关信息并生成回答"
    
    def _execute_react(self, query: str, plan: str, conversation_history: str = "") -> tuple:
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
    
    def _generate_final_answer(self, query: str, execution_process: List[Dict], retrieved_info: List[Dict], conversation_history: str = "") -> str:
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
                "retrieved_info": retrieved_info_str,
                "conversation_history": conversation_history
            })
            
            return answer
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return f"抱歉，生成回答时出现错误：{str(e)}"
    
    async def _generate_final_answer_stream(self, query: str, process_str: str, retrieved_info_str: str, conversation_history: str = "") -> List[str]:
        """流式生成最终回答"""
        try:
            # 生成回答
            summarize_chain = self.summarize_prompt | self.llm | self.output_parser
            answer = summarize_chain.invoke({
                "query": query,
                "execution_process": process_str,
                "retrieved_info": retrieved_info_str,
                "conversation_history": conversation_history
            })
            
            # 将回答分成小块
            chunks = []
            chunk_size = 20  # 每次返回20个字符
            
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                chunks.append(chunk)
            
            return chunks
        except Exception as e:
            logger.error(f"流式生成回答失败: {str(e)}")
            return [f"抱歉，生成回答时出现错误：{str(e)}"]
    
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
