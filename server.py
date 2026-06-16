# server.py
import sys
from pathlib import Path
import os
import json
import uuid
import asyncio

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_cors import CORS

from core.node import Flow
from core.memory import Memory
from core.nodes import (
    StartNode, ScaleRouterNode, QuestionNode, AnswerNode,
    ScoringNode, FollowUpRouterNode, ClassMatchNode, IEPNode, ReportNode,
    ChatNode, ToolCallNode, KnowledgeNode, OutputNode
)
from core.tools import SYSTEM_PROMPT, TOOLS
from core.assessment_store import AssessmentStore

app = Flask(__name__)
CORS(app)

# ================== 会话存储 ==================
sessions = {}
SESSIONS_FILE = "sessions.json"

def save_sessions():
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] 保存会话失败: {e}")

def load_sessions():
    global sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                sessions = json.load(f)
                print(f"[OK] 已恢复 {len(sessions)} 个聊天会话")
        except Exception as e:
            print(f"[WARN] 加载会话失败: {e}")

def get_or_create_chat_session(sid):
    if not sid or sid not in sessions:
        sid = f"chat_{uuid.uuid4().hex[:8]}"
        sessions[sid] = []
    return sid
# ================== 批量解析 ==================
def parse_user_answer(message, questions, current_qid, total):
    """
    通用答案解析，支持任意量表的选项配置。
    返回答案列表（元素为选项索引字符串 "0"/"1"/"2"/"3" 或 "yes"/"no"）。
    """
    import re
    msg = message.strip()
    
    if current_qid >= total:
        return []
    
    current_q = questions[current_qid] if current_qid < len(questions) else None
    if not current_q:
        return []
    
    options = current_q.get("options", [])
    
    # ===== 兼容旧版 M-CHAT-R（无 options，只有 yes/no）=====
    if not options:
        # 批量：如"全是是""都选否""统一yes"
        if re.search(r'(?:全[部是]|所有|统一|一律|都)\s*[选是]?[择为]?(是|否|yes|no)', msg, re.I):
            val = re.search(r'(?:全[部是]|所有|统一|一律|都)\s*[选是]?[择为]?(是|否|yes|no)', msg, re.I).group(1).lower()
            val = "yes" if val in ["是", "yes"] else "no"
            remaining = total - current_qid
            return [val] * remaining
        
        # 单题
        if "是" in msg or "yes" in msg.lower():
            return ["yes"]
        elif "否" in msg or "no" in msg.lower():
            return ["no"]
        return []
    
    # ===== 有 options 的量表 =====
    
    # 构建文本->索引映射（支持全称和前两字简写）
    text_to_idx = {}
    for i, opt in enumerate(options):
        label = opt["label"]
        text_to_idx[label] = str(i)
        if len(label) >= 2:
            text_to_idx[label[:2]] = str(i)  # 如"无此"->0,"有此"->1
    
    # 辅助：把用户输入的文本/数字转成选项索引
    def resolve_to_idx(text):
        text = text.strip()
        # 中文数字/阿拉伯数字（1-based）→ 0-based 索引
        num_map = {"一":1, "二":2, "三":3, "四":4, "1":1, "2":2, "3":3, "4":4}
        if text in num_map:
            idx = num_map[text] - 1
            if 0 <= idx < len(options):
                return str(idx)
        # 文本模糊匹配（如"无此表现""还算不少"）
        for label, idx in text_to_idx.items():
            if text in label or label in text:
                return idx
        return None
    
    # 1. 批量同答案："全是2""所有题都是无此表现""统一选第三个"
    batch_patterns = [
        r'(?:全[部是]|所有题?[目都]?|统一|一律|都)[选是]?[择为]?(.+)',
    ]
    for pattern in batch_patterns:
        m = re.search(pattern, msg)
        if m:
            idx = resolve_to_idx(m.group(1))
            if idx is not None:
                remaining = total - current_qid
                return [idx] * remaining
    
    # 2. 范围批量："第1-15题都是2""第1到15题都是无此表现"
    range_batch = re.search(r'(?:第?\s*)?(\d+)[\-~至到](\d+)(?:题?)?[都全]?[是]?(.+)', msg)
    if range_batch:
        start = int(range_batch.group(1)) - 1
        end = int(range_batch.group(2))
        idx = resolve_to_idx(range_batch.group(3))
        if idx is not None and start <= current_qid < end:
            count = min(end, total) - current_qid
            return [idx] * count
    
    # 3. 逗号/空格分隔批量："1,2,3,2,1" 或 "1 2 3 2 1"
    nums = re.findall(r'\b([1234])\b', msg)
    if nums and len(nums) > 1:
        remaining = total - current_qid
        idx_list = [str(int(n)-1) for n in nums]
        # 数量正好匹配，或用户明确声明是批量答案
        if len(idx_list) == remaining or any(k in msg for k in ['答案', '如下', '全部', '以下']):
            return idx_list[:remaining]
    
    # 4. 单题回答
    # 4.1 数字（1-based）
    if msg in ["1", "2", "3", "4"]:
        idx = int(msg) - 1
        if 0 <= idx < len(options):
            return [str(idx)]
    
    # 4.2 文本匹配（如"无此表现""还算不少"）
    for label, idx in text_to_idx.items():
        if label in msg or msg in label:
            return [idx]
    
    return []


def build_question_prompt(questions, current_qid, llm_questions):
    """根据当前题目动态生成提示，适配任意量表"""
    if current_qid >= len(questions):
        return "评估已完成。"
    
    q = questions[current_qid]
    lq = llm_questions[current_qid] if current_qid < len(llm_questions) else {
        "text": q["text"],
        "options": [o["label"] for o in q.get("options", [])]
    }
    
    total = len(questions)
    opts = lq.get("options", [])
    
    # 构建选项提示
    if not opts:
        # M-CHAT-R 旧版
        opts_str = "1. 是\n2. 否"
    else:
        opts_str = "\n".join([f"{i+1}. {o}" for i, o in enumerate(opts)])
    
    # 关键项标记
    critical_tag = "【关键项】" if q.get("critical") else ""
    
    return f"第 {current_qid+1} 题 / 共 {total} 题 {critical_tag}\n\n{lq['text']}\n\n{opts_str}\n\n💡 回复数字（如 1）或关键词（如 {opts[0] if opts else '是'}）即可"
# ================== 评估存储 ==================
assessment_store = AssessmentStore()

# ================== 知识库加载 ==================
def read_docx(path):
    from docx import Document
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def read_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)

def init_knowledge_base():
    import chromadb
    knowledge_base = []
    base_dir = Path(__file__).parent
    knowledge_dir = base_dir / "knowledge"

    if knowledge_dir.exists():
        for root, dirs, files in os.walk(str(knowledge_dir)):
            for file in files:
                path = os.path.join(root, file)
                category = os.path.basename(root)
                if file.endswith(".txt"):
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                elif file.endswith(".docx"):
                    content = read_docx(path)
                elif file.endswith(".pdf"):
                    content = read_pdf(path)
                else:
                    continue
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for i, p in enumerate(paragraphs):
                    knowledge_base.append({
                        "document": p,
                        "id": f"{category}_{file}_{i}",
                        "metadata": {"category": category, "source": file}
                    })

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="agent_docs")
    except:
        pass
    collection = chroma_client.create_collection(name="agent_docs")
    if knowledge_base:
        collection.add(
            documents=[k["document"] for k in knowledge_base],
            ids=[k["id"] for k in knowledge_base],
            metadatas=[k["metadata"] for k in knowledge_base]
        )
        print(f"[OK] 知识库已加载：{len(knowledge_base)} 段文本")
    else:
        print(f"[WARN] 未找到知识库文件，请检查 {knowledge_dir}")
    return collection

chroma_collection = None

# ================== 评估路由 ==================
@app.route('/assessment/start', methods=['POST'])
def assessment_start():
    """创建评估（通常由 Agent 工具调用触发，也可前端直接调用）"""
    try:
        data = request.json or {}
        scale = data.get('scale', 'mchat_r')
        asm_id = assessment_store.create(scale, {})
        return jsonify({
            "assessment_id": asm_id,
            "status": "created"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/assessment/<asm_id>/intake', methods=['POST'])
def assessment_intake(asm_id):
    print(f"[API] /assessment/{asm_id}/intake: child={data}")
    ctx = assessment_store.get(asm_id)
    if not ctx:
        return jsonify({"error": "评估不存在"}), 404

    try:
        data = request.json or {}
        StartNode.init(ctx, ctx["scale_name"])
        ctx["child"] = {
            "name": data.get("name", "未知"),
            "age": data.get("age", 0),
            "gender": data.get("gender", "未知")
        }
        ctx["concerns"] = data.get("concerns", "")
        ctx["status"] = "intake_done"

        # 路由选量表
        router = ScaleRouterNode()
        asyncio.run(router._exec(None, ctx))

        # 取第一个量表的第一题
        qnode = QuestionNode()
        action, q_data = asyncio.run(qnode._exec(None, ctx))

        assessment_store.update(asm_id, **ctx)
        return jsonify({
            "status": "waiting_answer",
            "question": q_data,
            "progress": q_data.get("progress") if q_data else None,
            "selected_scales": ctx.get("selected_scales", []),
            "current_scale": ctx.get("current_scale_name", "")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    
@app.route('/assessment/<asm_id>/answer', methods=['POST'])
def assessment_answer(asm_id):
    print(f"[API] /assessment/{asm_id}/answer: status={ctx.get('status')}, qid={ctx.get('current_qid')}")
    ctx = assessment_store.get(asm_id)
    if not ctx:
        return jsonify({"error": "评估不存在"}), 404

    if ctx.get("status") != "waiting_answer":
        if ctx.get("status") == "completed":
            return jsonify({"status": "completed", "report": ctx.get("report")})
        return jsonify({"status": ctx.get("status"), "question": ctx.get("current_question")})

    try:
        data = request.json or {}
        answer_val = data.get("answer")

        # 提交答案
        anode = AnswerNode()
        asyncio.run(anode._exec({"answer": answer_val}, ctx))

        # 取下一题
        qnode = QuestionNode()
        action, payload = asyncio.run(qnode._exec(None, ctx))

        if action == "done":
            # 当前量表完成，计分
            scoring = ScoringNode()
            asyncio.run(scoring._exec(None, ctx))

            # 判断是否追加量表
            follow = FollowUpRouterNode()
            action2, payload2 = asyncio.run(follow._exec(None, ctx))

            if action2 == "next_scale":
                assessment_store.update(asm_id, **ctx)
                return jsonify({
                    "status": "scale_change",
                    "message": payload2.get("message", ""),
                    "next_scale": payload2.get("next_scale", ""),
                    "question": ctx.get("current_question"),
                    "progress": ctx.get("current_question", {}).get("progress"),
                    "selected_scales": ctx.get("selected_scales", []),
                    "current_scale": ctx.get("current_scale_name", "")
                })

            # 所有量表完成，做分班匹配
            matcher = ClassMatchNode()
            asyncio.run(matcher._exec(None, ctx))
            # 生成 IEP（中高风险才生成）
            iep_node = IEPNode()
            asyncio.run(iep_node._exec(None, ctx))
            # 生成综合报告
            rnode = ReportNode()
            _, report_data = asyncio.run(rnode._exec(None, ctx))

            assessment_store.update(asm_id, **ctx)
            assessment_store.update(asm_id, session_id="")  # 解绑，防止后续聊天误判
            return jsonify({
                "status": "completed",
                "report": report_data
            })

        assessment_store.update(asm_id, **ctx)
        return jsonify({
            "status": "screening",
            "question": payload,
            "progress": payload.get("progress")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "系统内部错误"}), 500

@app.route('/assessment/<asm_id>/status', methods=['GET'])
def assessment_status(asm_id):
    print(f"[API] /assessment/{asm_id}/status: status={ctx.get('status')}")
    ctx = assessment_store.get(asm_id)
    if not ctx:
        return jsonify({"error": "评估不存在"}), 404

    resp = {
        "assessment_id": asm_id,
        "status": ctx.get("status"),
        "child": ctx.get("child"),
        "concerns": ctx.get("concerns", ""),
        "selected_scales": ctx.get("selected_scales", []),
        "current_scale": ctx.get("current_scale_name", ""),
        "progress": {
            "current": ctx.get("current_qid", 0),
            "total": ctx.get("total", 0)
        }
    }
    if ctx.get("status") == "waiting_answer":
        resp["question"] = ctx.get("current_question")
    if ctx.get("status") == "completed":
        resp["report"] = ctx.get("report")
    return jsonify(resp)

# ================== 聊天路由 ==================
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        user_message = data.get('message', '')
        session_id = data.get('session_id', '')

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        sid = get_or_create_chat_session(session_id)
        print(f"\n[API] /chat: session={sid}, msg='{user_message[:50]}...'")

        # ===== 评估快速通道 =====
                # ===== 评估快速通道 =====
        active_asm = None
        for asm_id, asm in assessment_store.all().items():
            if asm.get("session_id") == sid and asm.get("status") in ["waiting_answer", "screening"]:
                active_asm = asm_id
                break
        
        # 双重保险：已完成的评估绝不走快速通道
        if active_asm:
            asm_check = assessment_store.get(active_asm)
            if asm_check.get("status") == "completed":
                print(f"[API] /chat: asm={active_asm} already completed, skip fast path")
                active_asm = None
        
        print(f"[API] /chat: active_asm={active_asm}")

        if active_asm:
            asm = assessment_store.get(active_asm)
            questions = asm.get("questions", [])
            llm_questions = asm.get("llm_questions", [])
            current_qid = asm.get("current_qid", 0)
            total = len(questions)

            print(f"[API] /chat: fast_path, qid={current_qid}/{total}, status={asm.get('status')}")

            if current_qid >= total and asm.get("status") == "completed":
                print(f"[API] /chat: assessment already completed, switch to normal chat")
                # 评估已完成，走正常聊天
                pass
            else:
                # 解析用户输入
                answers = parse_user_answer(user_message, questions, current_qid, total)
                print(f"[API] /chat: parsed_answers={answers}")

                if not answers:
                    cq = current_qid if current_qid < total else total - 1
                    q_text = llm_questions[cq]["text"] if cq < len(llm_questions) else questions[cq]["text"]
                    opts = llm_questions[cq]["options"] if cq < len(llm_questions) else [o["label"] for o in questions[cq].get("options", [])]
                    opts_str = "\n".join([f"{i+1}. {o}" for i, o in enumerate(opts)])
                    return jsonify({
                        'reply': f"请按选项回答哦 😊\n\n第 {cq+1} 题 / 共 {total} 题\n\n{q_text}\n\n{opts_str}\n\n💡 提示：可以直接回复数字（如 3）或关键词（如 {opts[0] if opts else '是'}）",
                        'status': 'ok',
                        'session_id': sid
                    })

                # 批量提交答案
                for ans in answers:
                    anode = AnswerNode()
                    asyncio.run(anode._exec({"answer": ans}, asm))

                # 检查是否完成当前量表
                if asm.get("current_qid", 0) >= total:
                    print(f"[API] /chat: scale finished, scoring...")
                    scoring = ScoringNode()
                    asyncio.run(scoring._exec(None, asm))

                    follow = FollowUpRouterNode()
                    action2, payload2 = asyncio.run(follow._exec(None, asm))

                    if action2 == "next_scale":
                        print(f"[API] /chat: next_scale={payload2.get('next_scale')}")
                        assessment_store.update(active_asm, **asm)
                        tc_node = ToolCallNode()
                        asyncio.run(tc_node._pregenerate_questions(asm))
                        assessment_store.update(active_asm, **asm)

                        first_q = asm.get("llm_questions", [{}])[0]
                        opts_str = "\n".join([f"{i+1}. {o}" for i, o in enumerate(first_q.get("options", []))])
                        return jsonify({
                            'reply': f"{payload2.get('message', '')}\n\n第一题：{first_q.get('text', '')}\n\n{opts_str}",
                            'status': 'ok',
                            'session_id': sid
                        })

                    # 全部完成
                    print(f"[API] /chat: all scales finished, generating report...")
                    matcher = ClassMatchNode()
                    asyncio.run(matcher._exec(None, asm))

                    # 生成 IEP（中高风险才生成）
                    iep_node = IEPNode()
                    asyncio.run(iep_node._exec(None, asm))

                    rnode = ReportNode()
                    _, report_data = asyncio.run(rnode._exec(None, asm))
                    assessment_store.update(active_asm, **asm)
                    assessment_store.update(active_asm, session_id="")  # 解绑

                    # 清理历史中的评估工具消息，防止 LLM 400
                                        # 清理评估相关的消息，防止 LLM 400
                                        # 彻底清理评估痕迹，并插入系统提示告诉LLM评估已结束
                    cleaned = []
                    for msg in sessions[sid]:
                        # 删除所有评估相关的 tool 和 tool_calls
                        if msg.get("role") == "tool":
                            continue
                        if msg.get("role") == "assistant" and msg.get("tool_calls"):
                            continue
                        cleaned.append(msg)
                    
                    # 插入一条系统提示，强制LLM回到咨询模式
                    cleaned.append({
                        "role": "system",
                        "content": "评估已完成，用户已收到正式报告。现在回到正常咨询模式，回答家长关于报告解读、教育建议、学校政策等问题。严禁再次展示题目、询问选项或调用评估工具，除非用户明确说'重新评估'。"
                    })
                    
                    sessions[sid] = cleaned
                    save_sessions()

                    # 构建报告摘要，注入记忆
                    child = report_data.get('child', {})
                    scales = report_data.get('scales', [])
                    class_sug = report_data.get('class_suggestion', {})
                    risk_name = '🔴 高风险' if report_data.get('risk_level') == 'red' else '🟡 中风险' if report_data.get('risk_level') == 'yellow' else '🟢 低风险'
                    
                    scale_brief = ", ".join([
                        f"{s.get('scale_name', '')}: {s.get('risk_name', '')}" 
                        for s in scales
                    ])
                    
                    memory_summary = (
                        f"已为 {child.get('name', '未知')} 完成入学能力评估。"
                        f"量表结果：{scale_brief}。"
                        f"综合风险：{risk_name}。"
                        f"分班建议：{class_sug.get('class_type', '')}。"
                        f"家长主诉：{asm.get('concerns', '')}。"
                    )

                    # 注入到会话记忆（作为 assistant 消息）
                    sessions[sid].append({
                        "role": "assistant",
                        "content": memory_summary
                    })
                    save_sessions()

                    print(f"[API] /chat: assessment completed, injected memory summary")

                    return jsonify({
                        'reply': f"✅ 评估完成！\n\n{report_data.get('report_text', '')}\n\n【分班建议】{report_data.get('class_suggestion', {}).get('class_type', '')}",
                        'status': 'ok',
                        'session_id': sid
                    })
                
                                # 返回下一题（零延迟）—— 不保存到记忆，避免污染
                next_qid = asm.get("current_qid", 0)
                next_llm = llm_questions[next_qid] if next_qid < len(llm_questions) else {
                    "text": questions[next_qid]["text"],
                    "options": [o["label"] for o in questions[next_qid].get("options", [])]
                }
                opts_str = "\n".join([f"{i+1}. {o}" for i, o in enumerate(next_llm["options"])])
                assessment_store.update(active_asm, **asm)

                print(f"[API] /chat: next_qid={next_qid+1}, returning question, NOT saved to memory")

                # ⚠️ 关键：评估答题期间不写入 sessions，避免记忆污染
                # 用户答案和题目提示都不进入 LLM 历史

                return jsonify({
                    'reply': f"第 {next_qid+1} 题 / 共 {total} 题\n\n{next_llm['text']}\n\n{opts_str}",
                    'status': 'ok',
                    'session_id': sid
                })

        # ===== 正常聊天流 =====
        print(f"[API] /chat: normal chat path")
        messages = sessions[sid].copy()
        messages.append({"role": "user", "content": user_message})
        messages = memory.process(messages)

        chat_node = ChatNode()
        tool_call = ToolCallNode()
        knowledge = KnowledgeNode()
        output = OutputNode()

        chat_node - "tool_call" >> tool_call
        chat_node - "rag" >> knowledge
        chat_node - "assessment" >> tool_call
        tool_call - "chat" >> chat_node
        knowledge - "chat" >> chat_node
        chat_node - "output" >> output

        ctx = {
            "messages": messages,
            "tools": TOOLS,
            "system_prompt": SYSTEM_PROMPT,
            "chroma_collection": chroma_collection,
            "memory": memory,
            "assessment_store": assessment_store,
            "session_id": sid
        }

        flow = Flow(chat_node)
        last_action, payload = asyncio.run(flow.run(None, ctx))

        assistant_message = payload
        content = assistant_message.get("content", "") if isinstance(assistant_message, dict) else str(assistant_message)

        # 检测是否触发了评估创建
        assessment_info = None
        for msg in ctx["messages"]:
            if msg.get("role") == "tool":
                try:
                    tc_content = msg.get("content", "")
                    if "assessment_created" in tc_content:
                        assessment_info = json.loads(tc_content)
                except:
                    pass

        # 如果创建了评估，强制把第一题拼接到回复末尾
        if assessment_info and assessment_info.get("first_question"):
            first_q = assessment_info.get("first_question", {})
            llm_qs = assessment_info.get("llm_questions", [])
            q_text = llm_qs[0]["text"] if llm_qs and len(llm_qs) > 0 else first_q.get("text", "")
            opts = llm_qs[0]["options"] if llm_qs and len(llm_qs) > 0 else [o["label"] for o in first_q.get("options", [])]
            opts_str = "\n".join([f"{i+1}. {o}" for i, o in enumerate(opts)])
            scales = assessment_info.get("selected_scales", [])
            content += f"\n\n---\n\n🎓 特需儿童入学能力评估\n📋 评估组合：{' → '.join(scales)}\n\n第 1 题 / 共 {first_q.get('progress', {}).get('total', '?')} 题\n\n{q_text}\n\n{opts_str}\n\n💡 请直接回复数字（如 1）或关键词（如 {opts[0] if opts else '是'}）"

        sessions[sid] = ctx["messages"].copy()
        save_sessions()

        print(f"[API] /chat: normal reply, len={len(content)}")

        resp = {
            'reply': content,
            'status': 'ok',
            'session_id': sid
        }
        if assessment_info:
            resp["assessment"] = assessment_info

        return jsonify(resp)

    except Exception as e:
        import traceback
        traceback.print_exc()
        
        # 不返回 500，而是把异常信息塞给 LLM，让 LLM 组织语言回复
        try:
            error_desc = str(e)[:300]
            # 把异常作为 system 消息注入当前会话
            if sid in sessions:
                sessions[sid].append({
                    "role": "system",
                    "content": f"系统刚才遇到技术问题：{error_desc}。请用温暖专业的语言向用户说明情况，建议稍后重试或联系管理员，绝对不要输出技术错误代码。"
                })
                save_sessions()
            
            # 再次调用 LLM，让它基于异常信息生成友好回复
            retry_messages = sessions.get(sid, []).copy()
            retry_messages = memory.process(retry_messages)
            
            from core.llm import call_llm_simple
            friendly_reply = call_llm_simple(
                f"你是一位特殊教育助手。系统刚才出了点问题：{error_desc}。请用一句话向用户道歉并安抚，建议刷新页面或稍后重试。语气温暖，不要提技术细节。"
            )
            
            return jsonify({
                'reply': friendly_reply,
                'status': 'ok',
                'session_id': sid
            })
            
        except Exception as fallback_error:
            # 如果连 fallback 都失败了，返回最简化的兜底消息
            return jsonify({
                'reply': '抱歉，系统刚才有点小状况，已经恢复啦。如果问题持续，请刷新页面或稍后再试 😊',
                'status': 'ok',
                'session_id': sid
            })

@app.route('/sessions', methods=['GET'])
def list_sessions():
    return jsonify({'sessions': list(sessions.keys())})

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    if session_id not in sessions:
        return jsonify({'error': '会话不存在'}), 404
    return jsonify({'session_id': session_id, 'messages': sessions[session_id]})

@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in sessions:
        del sessions[session_id]
        save_sessions()
        return jsonify({'status': 'deleted'})
    return jsonify({'error': '会话不存在'}), 404

@app.route('/clear', methods=['POST'])
def clear():
    data = request.json or {}
    session_id = data.get('session_id', '')
    if session_id and session_id in sessions:
        sessions[session_id] = []
        save_sessions()
        return jsonify({'status': 'cleared'})
    return jsonify({'status': 'error', 'message': '会话不存在'})

# ================== 启动 ==================
if __name__ == '__main__':
    load_sessions()
    chroma_collection = init_knowledge_base()
    memory = Memory(max_tokens=4000, keep_recent_tokens=2000)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("KIMICODE_API_KEY"):
        print("⚠️ 警告：未设置 API Key")

    print("=" * 55)
    print("🎓 EduScreenAgent Server")
    print("=" * 55)
    print("聊天模式 + 评估工具（按需调用）")
    print("=" * 55)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)