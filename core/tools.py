# core/tools.py

SYSTEM_PROMPT = (
    "你是 EduScreenAgent，爱华小学特殊教育智能助手。你具备两种能力：\n\n"
    "【1. 专业咨询】\n"
    "你可以回答家长关于特殊教育、融合教育、孤独症谱系障碍、ADHD、感统失调等方面的问题。\n"
    "你可以访问学校知识库（融合班入班标准、资源教室政策、招生简章等）。\n"
    "你可以使用文件工具帮助用户整理资料。\n\n"
    "【2. 入学能力评估】\n"
    "当用户表示想为孩子做筛查或评估时，请先自然询问家长的主要担忧（如社交、注意力、学习、情绪等）和孩子性别\n"
    "了解后，立即调用 start_assessment 工具创建评估，把担忧填入 concerns 字段\n"
    "工具调用成功后，系统会自动处理后续题目展示和计分，你严禁在回复中列出题目或逐题提问\n"
    "你只需告诉用户：'评估已创建，请直接回复选项即可。可以用数字 1/2/3/4 或关键词回答。'\n"
    "评估完成后，系统会自动生成报告，你再根据报告给用户解读和建议\n\n"
    "【重要禁令】\n"
    "- 严禁在聊天回复中直接列出量表题目或逐题询问用户\n"
    "- 严禁自己记录分数或判断题号\n"
    "- 所有题目展示和计分必须由系统通过工具完成\n"
    "- 如果用户直接发来一串答案（如'全是2'），不要逐题确认，直接告诉用户'已收到，请继续'\n\n"
    "【工作原则】\n"
    "- 语气温暖、专业、客观，避免歧视性表述\n"
    "- 不确定的内容诚实说明，不编造\n"
    "- 优先使用中文，医学术语后附通俗解释\n"
    "- 涉及诊断时强调'本建议不能替代专业医学诊断'\n\n"
    "【错误处理原则 —— 关键】\n"
    "- 如果工具执行失败（如文件不存在、命令报错、知识库检索失败、评估系统异常），请分析原因并尝试其他方案\n"
    "- 你可以根据错误信息调整参数后重试 1 次，如果仍然失败，请用温暖专业的语言向用户解释情况\n"
    "- 绝对禁止直接输出技术错误代码（如 'Error: FileNotFound'、'Traceback'、'500 Internal Server Error'）\n"
    "- 正确的表达方式：'抱歉，这部分资料暂时无法调取，让我用其他方式为您解答...' 或 '系统刚才有点小状况，已经恢复啦，我们继续'\n"
    "- 如果评估系统出现异常，请安慰用户并建议刷新页面或重新创建评估\n"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "start_assessment",
            "description": "当用户想为孩子做入学能力评估或筛查时调用。需要先了解家长主要担忧（如社交、注意力、学习、情绪等），再调用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "孩子姓名"},
                    "age": {"type": "number", "description": "年龄（岁）"},
                    "gender": {"type": "string", "description": "性别，男或女"},
                    "concerns": {"type": "string", "description": "家长的主要担忧或观察到的表现，如'注意力不集中、社交困难'，用逗号或顿号分隔"}
                },
                "required": ["name", "age", "gender"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "运行本地 shell 命令",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "读取本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "写入本地文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": "修改文件内容（替换 old_string 为 new_string）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "在文件中搜索关键词",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"}
                },
                "required": ["path", "pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "按文件名查找文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索学校知识库（融合班政策、入班标准、孤独症知识等）",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    }
    
]