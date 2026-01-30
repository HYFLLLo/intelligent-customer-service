import os
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from loguru import logger

class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, memory_file: str = "./data/memory/conversations.json", max_history: int = 10, max_memory_hours: int = 24):
        """初始化对话记忆管理器"""
        self.memory_file = memory_file
        self.max_history = max_history  # 每个会话最多保存的消息数
        self.max_memory_hours = max_memory_hours  # 记忆最长保存时间（小时）
        
        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        
        # 加载现有记忆
        self.conversations = self._load_memory()
        logger.info(f"对话记忆管理器初始化完成，加载了 {len(self.conversations)} 个会话")
    
    def _load_memory(self) -> Dict[str, List[Dict]]:
        """加载对话记忆"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8-sig') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载对话记忆失败: {str(e)}")
        return {}
    
    def _save_memory(self):
        """保存对话记忆"""
        try:
            # 清理过期记忆
            self._clean_expired_memory()
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            with open(self.memory_file, 'w', encoding='utf-8-sig') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"保存对话记忆失败: {str(e)}")
    
    def _clean_expired_memory(self):
        """清理过期记忆"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, messages in self.conversations.items():
            if not messages:
                expired_sessions.append(session_id)
                continue
            
            # 检查最后一条消息的时间
            last_message_time = datetime.fromisoformat(messages[-1]['timestamp'])
            if (current_time - last_message_time).total_seconds() > self.max_memory_hours * 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.conversations[session_id]
            logger.info(f"清理过期会话: {session_id}")
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加对话消息"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # 添加新消息
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        self.conversations[session_id].append(message)
        
        # 保持消息数量在限制内
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]
        
        # 保存记忆
        self._save_memory()
        logger.debug(f"添加消息到会话 {session_id}: {role} - {content[:50]}...")
    
    def get_conversation_history(self, session_id: str, max_messages: Optional[int] = None) -> List[Dict]:
        """获取对话历史"""
        if session_id not in self.conversations:
            return []
        
        history = self.conversations[session_id]
        if max_messages and len(history) > max_messages:
            return history[-max_messages:]
        return history
    
    def get_context(self, session_id: str, max_messages: int = 5) -> str:
        """获取对话上下文"""
        history = self.get_conversation_history(session_id, max_messages)
        if not history:
            return ""
        
        context = "对话历史：\n"
        for message in history:
            role = "用户" if message['role'] == 'user' else "系统"
            context += f"{role}: {message['content']}\n"
        
        return context
    
    def get_personal_info(self, session_id: str) -> str:
        """获取用户个人信息"""
        history = self.get_conversation_history(session_id)
        if not history:
            return ""
        
        personal_info = "用户个人信息：\n"
        for message in history:
            if message['role'] == 'user':
                content = message['content']
                # 检查是否包含个人信息
                if any(keyword in content for keyword in ['名字', '叫什么', '年龄', '多大', '岁']):
                    personal_info += f"{content}\n"
        
        return personal_info if personal_info != "用户个人信息：\n" else ""
    
    def clear_session(self, session_id: str):
        """清除指定会话的记忆"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            self._save_memory()
            logger.info(f"清除会话记忆: {session_id}")
    
    def clear_all(self):
        """清除所有对话记忆"""
        self.conversations = {}
        self._save_memory()
        logger.info("清除所有对话记忆")
