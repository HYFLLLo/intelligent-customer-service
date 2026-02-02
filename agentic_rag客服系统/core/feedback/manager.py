import os
import json
import datetime
from typing import List, Dict, Optional
from collections import Counter
import pandas as pd
from loguru import logger

class FeedbackManager:
    def __init__(self):
        """初始化反馈管理器"""
        self.feedback_dir = os.getenv("FEEDBACK_PATH", "./data/feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)
        
        # 反馈数据文件
        self.feedback_file = os.path.join(self.feedback_dir, "feedback.json")
        
        # 初始化反馈数据
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
        
        # 高频问题阈值
        self.high_freq_threshold = 3
        
        logger.info("反馈管理器初始化完成")
    
    def add_feedback(self, query_id: str, is_solved: bool, additional_info: str = ""):
        """添加用户反馈"""
        try:
            # 加载现有反馈
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            
            # 添加新反馈
            new_feedback = {
                "query_id": query_id,
                "is_solved": is_solved,
                "additional_info": additional_info,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            feedbacks.append(new_feedback)
            
            # 保存反馈
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump(feedbacks, f, ensure_ascii=False, indent=2)
            
            logger.info(f"反馈添加成功: {query_id}, 解决状态: {is_solved}")
            
            # 分析反馈
            self.analyze_feedback()
            
            return True
        except Exception as e:
            logger.error(f"添加反馈失败: {str(e)}")
            return False
    
    def analyze_feedback(self):
        """分析用户反馈"""
        try:
            # 加载反馈数据
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            
            if not feedbacks:
                logger.warning("反馈数据为空")
                return
            
            # 分析未解决问题
            unsolved_feedbacks = [f for f in feedbacks if not f.get("is_solved", True)]
            
            if unsolved_feedbacks:
                # 提取未解决问题的补充信息
                additional_infos = [f.get("additional_info", "") for f in unsolved_feedbacks if f.get("additional_info", "")]
                
                if additional_infos:
                    # 分析高频问题
                    self.identify_high_freq_issues(additional_infos)
            
            # 生成反馈分析报告
            self.generate_feedback_report(feedbacks)
            
        except Exception as e:
            logger.error(f"分析反馈失败: {str(e)}")
    
    def identify_high_freq_issues(self, additional_infos: List[str]):
        """识别高频问题"""
        try:
            # 简单的关键词提取和统计
            keywords = []
            for info in additional_infos:
                # 简单分词（实际应用中可使用更复杂的NLP工具）
                words = [w for w in info.split() if len(w) > 1]
                keywords.extend(words)
            
            # 统计关键词频率
            keyword_counts = Counter(keywords)
            high_freq_keywords = [k for k, v in keyword_counts.items() if v >= self.high_freq_threshold]
            
            if high_freq_keywords:
                logger.info(f"识别到高频问题关键词: {high_freq_keywords}")
                # 生成优化建议
                self.generate_optimization_suggestions(high_freq_keywords)
        except Exception as e:
            logger.error(f"识别高频问题失败: {str(e)}")
    
    def generate_optimization_suggestions(self, high_freq_keywords: List[str]):
        """生成优化建议"""
        try:
            suggestions = {
                "high_freq_keywords": high_freq_keywords,
                "suggestions": [
                    f"建议在知识库中添加关于 '{keyword}' 的详细信息" for keyword in high_freq_keywords
                ],
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # 保存优化建议
            suggestions_file = os.path.join(self.feedback_dir, "optimization_suggestions.json")
            with open(suggestions_file, "w", encoding="utf-8") as f:
                json.dump(suggestions, f, ensure_ascii=False, indent=2)
            
            logger.info("优化建议生成完成")
        except Exception as e:
            logger.error(f"生成优化建议失败: {str(e)}")
    
    def generate_feedback_report(self, feedbacks: List[Dict]):
        """生成反馈分析报告"""
        try:
            # 计算解决率
            total_feedbacks = len(feedbacks)
            solved_count = sum(1 for f in feedbacks if f.get("is_solved", False))
            solved_rate = (solved_count / total_feedbacks * 100) if total_feedbacks > 0 else 0
            
            # 按时间分组分析
            df = pd.DataFrame(feedbacks)
            if not df.empty:
                df["date"] = pd.to_datetime(df["timestamp"]).dt.date
                daily_stats = df.groupby("date").agg(
                    total=pd.NamedAgg(column="is_solved", aggfunc="count"),
                    solved=pd.NamedAgg(column="is_solved", aggfunc="sum")
                ).reset_index()
                daily_stats["solved_rate"] = (daily_stats["solved"] / daily_stats["total"] * 100).round(2)
                daily_stats["date"] = daily_stats["date"].astype(str)
            else:
                daily_stats = pd.DataFrame()
            
            # 生成报告
            report = {
                "total_feedbacks": total_feedbacks,
                "solved_count": solved_count,
                "solved_rate": round(solved_rate, 2),
                "daily_stats": daily_stats.to_dict(orient="records"),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # 保存报告
            report_file = os.path.join(self.feedback_dir, "feedback_report.json")
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"反馈分析报告生成完成，解决率: {solved_rate:.2f}%")
        except Exception as e:
            logger.error(f"生成反馈报告失败: {str(e)}")
    
    def get_feedback_stats(self) -> Dict:
        """获取反馈统计信息"""
        try:
            # 加载反馈数据
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            
            if not feedbacks:
                return {
                    "total_feedbacks": 0,
                    "solved_count": 0,
                    "solved_rate": 0,
                    "unsolved_count": 0
                }
            
            # 计算统计信息
            total_feedbacks = len(feedbacks)
            solved_count = sum(1 for f in feedbacks if f.get("is_solved", False))
            unsolved_count = total_feedbacks - solved_count
            solved_rate = (solved_count / total_feedbacks * 100) if total_feedbacks > 0 else 0
            
            return {
                "total_feedbacks": total_feedbacks,
                "solved_count": solved_count,
                "solved_rate": round(solved_rate, 2),
                "unsolved_count": unsolved_count
            }
        except Exception as e:
            logger.error(f"获取反馈统计失败: {str(e)}")
            return {"error": str(e)}
    
    def suggest_knowledge_base_updates(self) -> List[str]:
        """建议知识库更新"""
        try:
            # 加载优化建议
            suggestions_file = os.path.join(self.feedback_dir, "optimization_suggestions.json")
            if not os.path.exists(suggestions_file):
                return []
            
            with open(suggestions_file, "r", encoding="utf-8") as f:
                suggestions = json.load(f)
            
            return suggestions.get("suggestions", [])
        except Exception as e:
            logger.error(f"建议知识库更新失败: {str(e)}")
            return []
    
    def clear_feedback(self):
        """清空反馈数据"""
        try:
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            
            logger.info("反馈数据已清空")
            return True
        except Exception as e:
            logger.error(f"清空反馈数据失败: {str(e)}")
            return False
    
    def get_all_feedbacks(self) -> List[Dict]:
        """获取所有反馈详情"""
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            
            # 按时间倒序排列
            feedbacks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return feedbacks
        except Exception as e:
            logger.error(f"获取所有反馈失败: {str(e)}")
            return []
    
    def get_feedbacks_by_status(self, is_solved: bool) -> List[Dict]:
        """按解决状态获取反馈"""
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            
            # 过滤指定状态的反馈
            filtered_feedbacks = [f for f in feedbacks if f.get("is_solved", False) == is_solved]
            
            # 按时间倒序排列
            filtered_feedbacks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return filtered_feedbacks
        except Exception as e:
            logger.error(f"按状态获取反馈失败: {str(e)}")
            return []
    
    def get_feedback_details(self, limit: int = 50) -> Dict:
        """获取详细的反馈统计信息"""
        try:
            with open(self.feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
            
            if not feedbacks:
                return {
                    "total_feedbacks": 0,
                    "solved_count": 0,
                    "solved_rate": 0,
                    "unsolved_count": 0,
                    "solved_feedbacks": [],
                    "unsolved_feedbacks": []
                }
            
            # 计算统计信息
            total_feedbacks = len(feedbacks)
            solved_count = sum(1 for f in feedbacks if f.get("is_solved", False))
            unsolved_count = total_feedbacks - solved_count
            solved_rate = (solved_count / total_feedbacks * 100) if total_feedbacks > 0 else 0
            
            # 按状态分组反馈
            solved_feedbacks = [f for f in feedbacks if f.get("is_solved", False)]
            unsolved_feedbacks = [f for f in feedbacks if not f.get("is_solved", False)]
            
            # 按时间倒序排列
            solved_feedbacks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            unsolved_feedbacks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # 限制返回数量
            solved_feedbacks = solved_feedbacks[:limit]
            unsolved_feedbacks = unsolved_feedbacks[:limit]
            
            return {
                "total_feedbacks": total_feedbacks,
                "solved_count": solved_count,
                "solved_rate": round(solved_rate, 2),
                "unsolved_count": unsolved_count,
                "solved_feedbacks": solved_feedbacks,
                "unsolved_feedbacks": unsolved_feedbacks
            }
        except Exception as e:
            logger.error(f"获取详细反馈统计失败: {str(e)}")
            return {"error": str(e)}
