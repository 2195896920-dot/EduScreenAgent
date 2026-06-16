# core/nodes.py
import json
import os
import asyncio
import re
import fnmatch
import math
from pathlib import Path
from typing import Any, Dict
from core.node import Node
from core.llm import call_llm_simple, call_llm


def log_node(node_name, action, payload=None, ctx_keys=None):
    """统一日志格式"""
    prefix = f"[NODE] {node_name}"
    if action:
        prefix += f" -> action='{action}'"
    if payload is not None:
        preview = str(payload)[:200]
        prefix += f" | payload_preview={preview}"
    if ctx_keys:
        prefix += f" | ctx_keys={ctx_keys}"
    print(prefix)


def load_scale(scale_name: str) -> Dict:
    path = Path(__file__).parent.parent / "scales" / f"{scale_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"量表文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================== 评估节点 ==================

class StartNode(Node):
    @staticmethod
    def init(ctx: Dict, scale_name: str):
        ctx["scale_name"] = scale_name
        ctx["answers"] = []
        ctx["current_qid"] = 0
        ctx["child"] = ctx.get("child", {})
        ctx["report"] = None
        ctx["status"] = "created"
        print(f"[ASSESS] StartNode.init: scale={scale_name}")


class IntakeNode(Node):
    async def exec(self, payload, ctx):
        ctx["child"] = payload
        ctx["status"] = "intake_done"
        print(f"[ASSESS] IntakeNode: child={payload}")
        return "next", None


class ScaleRouterNode(Node):
    async def exec(self, payload, ctx):
        age = ctx["child"].get("age", 0)
        concerns_text = ctx.get("concerns", "")
        concerns = [c.strip() for c in concerns_text.replace("，", ",").replace("、", ",").split(",") if c.strip()]
        print(f"[ASSESS] ScaleRouterNode: age={age}, concerns={concerns}")

        scale_map = {
            "社交": ["mchat_r", "abc", "srs_sf", "cars"],
            "沟通": ["mchat_r", "abc", "srs_sf", "cars"],
            "语言": ["mchat_r", "abc", "srs_sf"],
            "注意力": ["snap_iv"],
            "多动": ["snap_iv"],
            "冲动": ["snap_iv"],
            "学习困难": ["srs_sf", "snap_iv"],
            "学习": ["srs_sf", "snap_iv"],
            "情绪": ["snap_iv"],
            "行为": ["abc", "cars", "snap_iv"],
            "情绪行为": ["snap_iv", "cars"],
            "对立": ["snap_iv"],
            "违抗": ["snap_iv"],
            "全面": ["mchat_r", "abc", "snap_iv", "srs_sf", "cars"],
            "筛查": ["mchat_r", "abc", "snap_iv", "srs_sf", "cars"],
            "自闭症": ["mchat_r", "abc", "cars", "aq10"],
            "孤独症": ["mchat_r", "abc", "cars", "aq10"],
            "ASD": ["mchat_r", "abc", "cars", "aq10"],
            "刻板": ["abc", "cars"],
            "感统": ["abc"],
            "感知觉": ["abc"],
            "发育": ["mchat_r", "abc", "cars"],
            "迟缓": ["mchat_r", "abc", "cars"],
            "m-chat": ["mchat_r"],
            "abc": ["abc"],
            "srs": ["srs_sf"],
            "cars": ["cars"],
            "aq": ["aq10"],
            "snap": ["snap_iv"]
        }

        selected = []
        for c in concerns:
            for key, scales in scale_map.items():
                if key in c:
                    selected.extend(scales)
                    break

        age_filtered = []
        for s in selected:
            try:
                scale = load_scale(s)
                age_range = scale.get("age_range", [0, 99])
                if age_range[0] <= age <= age_range[1]:
                    age_filtered.append(s)
            except:
                pass

        seen = set()
        final = []
        for s in age_filtered:
            if s not in seen:
                seen.add(s)
                final.append(s)

        if not final:
            if 1.3 <= age <= 2.5:
                final = ["mchat_r"]
            elif 2 <= age <= 15:
                final = ["cars"]
            elif 4 <= age <= 18:
                final = ["srs_sf"]
            elif 6 <= age <= 13:
                final = ["snap_iv"]
            else:
                final = ["cars"]

        first_scale_name = final[0]
        first_scale = load_scale(first_scale_name)

        ctx["selected_scales"] = final
        ctx["current_scale_index"] = 0
        ctx["current_scale_name"] = first_scale_name
        ctx["all_reports"] = []
        ctx["scale"] = first_scale
        ctx["questions"] = first_scale["questions"]
        ctx["total"] = len(first_scale["questions"])
        ctx["answers"] = []
        ctx["current_qid"] = 0

        print(f"[ASSESS] ScaleRouterNode: selected={final}, first={first_scale_name}")
        return "next", {"selected_scales": final, "current": first_scale_name}


class QuestionNode(Node):
    async def exec(self, payload, ctx):
        qid = ctx["current_qid"]
        questions = ctx["questions"]

        if qid >= len(questions):
            print(f"[ASSESS] QuestionNode: all questions done ({qid}/{len(questions)})")
            return "done", None

        q = questions[qid]
        question_data = {
            "id": q["id"],
            "text": q["text"],
            "is_critical": q.get("critical", False),
            "options": q.get("options"),
            "progress": {"current": qid + 1, "total": len(questions)}
        }
        ctx["status"] = "waiting_answer"
        ctx["current_question"] = question_data
        print(f"[ASSESS] QuestionNode: qid={qid+1}/{len(questions)}, text={q['text'][:30]}...")
        return "wait_answer", question_data


class AnswerNode(Node):
    async def exec(self, payload, ctx):
        answer = payload.get("answer")
        qid = ctx["current_qid"]
        questions = ctx["questions"]

        if qid >= len(questions):
            print(f"[ASSESS] AnswerNode: already finished, skip")
            return "next", {"qid": None, "score": 0, "answer": answer}

        q = questions[qid]
        if "score_yes" in q:
            score = q["score_yes"] if answer == "yes" else q["score_no"]
        elif "score_map" in q:
            key = "yes" if answer == "yes" else "no"
            score = q["score_map"].get(key, 0)
        else:
            options = q.get("options", [])
            idx = int(answer) if answer is not None else 0
            if 0 <= idx < len(options):
                score = options[idx]["score"]
            else:
                score = 0

        ctx["answers"].append(score)
        ctx["current_qid"] += 1
        ctx["status"] = "screening"
        print(f"[ASSESS] AnswerNode: qid={q['id']}, answer={answer}, score={score}")
        return "next", {"qid": q["id"], "score": score, "answer": answer}


class ScoringNode(Node):
    async def exec(self, payload, ctx):
        scale = ctx["scale"]
        answers = ctx["answers"]
        questions = ctx["questions"]
        scoring = scale.get("scoring", {})
        method = scoring.get("method", "simple")

        # 统一计算 total，避免各分支重复定义或遗漏
        total = sum(answers)
        result = {"scale_name": scale.get("name", ""), "total": total, "method": method}

        if method == "subscale_threshold":
            subscales = scoring.get("subscales", {})
            positive_count = 0
            subscale_results = {}
            for name, cfg in subscales.items():
                indices = cfg.get("indices", [])
                threshold = cfg.get("threshold", 0)
                score_type = cfg.get("score_type", "gte2_count")

                if score_type == "gte2_count":
                    count = sum(1 for i in indices if i < len(answers) and answers[i] >= 2)
                else:
                    count = sum(answers[i] for i in indices if i < len(answers))

                met = count >= threshold
                if met:
                    positive_count += 1
                subscale_results[name] = {
                    "count": count,
                    "threshold": threshold,
                    "met": met,
                    "label": cfg.get("label", name)
                }

            result["subscales"] = subscale_results
            result["positive_subscales"] = positive_count

            levels = scoring.get("risk_levels", {})
            if positive_count >= 2:
                risk = "red"
                risk_name = levels.get("red", {}).get("label", "高风险")
            elif positive_count >= 1:
                risk = "yellow"
                risk_name = levels.get("yellow", {}).get("label", "中风险")
            else:
                risk = "green"
                risk_name = levels.get("green", {}).get("label", "低风险")

        elif method == "total_sum_multiplier":
            multiplier = scoring.get("multiplier", 1)
            rounding = scoring.get("rounding", "floor")
            standard = math.floor(total * multiplier) if rounding == "floor" else round(total * multiplier)
            result["standard_score"] = standard
            result["total"] = total

            levels = scoring.get("risk_levels", {})
            if standard >= 70:
                risk, risk_name = "red", levels.get("red", {}).get("label", "重度异常")
            elif standard >= 60:
                risk, risk_name = "orange", levels.get("orange", {}).get("label", "中度异常")
            elif standard >= 50:
                risk, risk_name = "yellow", levels.get("yellow", {}).get("label", "轻度异常")
            else:
                risk, risk_name = "green", levels.get("green", {}).get("label", "正常")

        elif method == "total_with_frequency":
            often_count = sum(1 for a in answers if a >= 2)
            result["total"] = total
            result["often_count"] = often_count

            levels = scoring.get("risk_levels", {})
            if total >= 14 and often_count >= 6:
                risk, risk_name = "red", levels.get("red", {}).get("label", "有孤独症倾向")
            elif total >= 7:
                risk, risk_name = "yellow", levels.get("yellow", {}).get("label", "可能有孤独症倾向")
            else:
                risk, risk_name = "green", levels.get("green", {}).get("label", "无孤独症倾向")

        else:
            # 默认简单计分（M-CHAT-R / ABC 改良版）
            critical_ids = [q["id"] for q in questions if q.get("critical")]
            critical_positive = sum(1 for cid in critical_ids if answers[cid - 1] > 0)
            result["critical_positive"] = critical_positive

            levels = scoring.get("risk_levels", scoring)
            red_cfg = levels.get("red") or {}
            yellow_cfg = levels.get("yellow") or {}
            green_cfg = levels.get("green") or {}
            red_min = red_cfg.get("min_score", 999)
            yellow_min = yellow_cfg.get("min_score", 999)

            if critical_positive >= 2 or total >= red_min:
                risk, risk_name = "red", red_cfg.get("label", "🔴 高风险")
            elif total >= yellow_min:
                risk, risk_name = "yellow", yellow_cfg.get("label", "🟡 中风险")
            else:
                risk, risk_name = "green", green_cfg.get("label", "🟢 低风险")

        result["risk_level"] = risk
        result["risk_name"] = risk_name
        ctx["scores"] = result
        ctx["risk_level"] = risk
        ctx["risk_name"] = risk_name
        ctx["status"] = "scoring"
        print(f"[ASSESS] ScoringNode: scale={scale.get('name')}, total={total}, risk={risk}")
        return "next", result


class FollowUpRouterNode(Node):
    async def exec(self, payload, ctx):
        reports = ctx.get("all_reports", [])
        selected = ctx.get("selected_scales", [])
        idx = ctx.get("current_scale_index", 0)

        if ctx.get("scores"):
            reports.append(ctx["scores"])
            ctx["all_reports"] = reports

        if idx + 1 < len(selected):
            ctx["current_scale_index"] = idx + 1
            next_scale_name = selected[idx + 1]
            ctx["current_scale_name"] = next_scale_name
            next_scale = load_scale(next_scale_name)
            ctx["scale"] = next_scale
            ctx["questions"] = next_scale["questions"]
            ctx["total"] = len(next_scale["questions"])
            ctx["answers"] = []
            ctx["current_qid"] = 0
            ctx["report"] = None
            print(f"[ASSESS] FollowUpRouterNode: next_scale={next_scale_name}")
            return "next_scale", {"next_scale": next_scale_name, "message": f"接下来进行 {next_scale.get('name', next_scale_name)} 评估"}

        print(f"[ASSESS] FollowUpRouterNode: all scales done, reports={len(reports)}")
        return "done", {"reports": reports}


class ClassMatchNode(Node):
    async def exec(self, payload, ctx):
        reports = ctx.get("all_reports", [])
        child = ctx.get("child", {})
        age = child.get("age", 0)
        has_red = any(r.get("risk_level") == "red" for r in reports)
        has_yellow = any(r.get("risk_level") in ["yellow", "orange"] for r in reports)

        attention_issue = social_issue = behavior_issue = False
        for r in reports:
            name = r.get("scale_name", "")
            if name == "SNAP-IV":
                subs = r.get("subscales", {})
                if subs.get("attention", {}).get("met") or subs.get("hyperactivity", {}).get("met"):
                    attention_issue = True
                if subs.get("oppositional", {}).get("met"):
                    behavior_issue = True
            elif name in ["SRS-SF", "CARS", "ABC-改良", "AQ-10"]:
                if r.get("risk_level") in ["yellow", "orange", "red"]:
                    social_issue = True

        if has_red:
            asd_red = any(r.get("scale_name") in ["CARS", "ABC-改良", "M-CHAT-R", "AQ-10"] and r.get("risk_level") == "red" for r in reports)
            if asd_red:
                suggestion = {"level": "red", "class_type": "建议暂缓入学，优先就医评估", "arrangement": "暂缓入学", "reason": "孤独症谱系筛查显示高风险信号，需专业医学诊断后再制定教育计划", "support": "建议尽快前往儿童发育行为科或儿童精神科就诊，早期干预对改善预后有重要意义"}
            else:
                suggestion = {"level": "yellow", "class_type": "全天融合 + 资源教室强化支持", "arrangement": "全天融合", "reason": "注意力/行为筛查阳性，提示 ADHD 可能，但认知能力可适应普通课堂，需额外行为支持", "support": "尽快就医确诊；学校提供：前排座位、任务分解、课间运动释放、每周2次资源教室注意力训练；家庭保持规律作息、正向鼓励"}

        elif social_issue and attention_issue:
            suggestion = {"level": "yellow", "class_type": "半天融合 + 资源教室强化支持", "arrangement": "半天融合", "reason": "社交与注意力均存在明显困难，需要较高强度的个别化支持", "support": "上午在普通班参与集体活动，下午在资源教室进行社交小组与注意力训练"}
        elif social_issue:
            suggestion = {"level": "yellow", "class_type": "全天融合 + 每周社交小组课" if age >= 6 else "融合班 + 学前社交干预", "arrangement": "全天融合" if age >= 6 else "融合班", "reason": "存在社交沟通困难，但可在普通课堂学习，需额外社交支持" if age >= 6 else "学前阶段社交困难，建议尽早干预", "support": "配备同伴支持，每周2次资源教室社交小组训练，视觉提示辅助" if age >= 6 else "影子老师辅助，感统训练，社交游戏小组"}
        elif attention_issue:
            suggestion = {"level": "yellow", "class_type": "全天融合 + 资源教室注意力训练" if age >= 6 else "融合班 + 感统与行为训练", "arrangement": "全天融合" if age >= 6 else "融合班", "reason": "注意力/多动问题明显，但认知能力可适应普通课堂" if age >= 6 else "学龄前多动明显，需行为管理与感统支持", "support": "座位安排在前排，分段任务，课间运动释放，每周2次资源教室注意力训练" if age >= 6 else "感统训练，正性行为支持，结构化日程"}
        elif has_yellow:
            suggestion = {"level": "yellow", "class_type": "融合班 + 定期观察", "arrangement": "融合班", "reason": "存在轻度风险，可在普通班学习并持续观察", "support": "教师每月记录行为表现，必要时转介资源教室"}
        else:
            suggestion = {"level": "green", "class_type": "全天融合班", "arrangement": "全天融合", "reason": "筛查结果正常，具备在普通班级学习的基本能力", "support": "正常参与班级活动，建议每学期常规发展监测"}

        ctx["class_suggestion"] = suggestion
        print(f"[ASSESS] ClassMatchNode: suggestion={suggestion['class_type']}, level={suggestion['level']}")
        return "next", suggestion

class IEPNode(Node):
    """根据评估结果生成本个性化教育计划（IEP）初稿"""
    async def exec(self, payload, ctx):
        child = ctx.get("child", {})
        reports = ctx.get("all_reports", [])
        class_suggestion = ctx.get("class_suggestion", {})
        risk_level = class_suggestion.get("level", "green")

        # 低风险无需 IEP
        if risk_level == "green":
            ctx["iep"] = None
            print(f"[ASSESS] IEPNode: risk=green, skip IEP generation")
            return "next", None

        # 构建量表摘要
        scale_lines = []
        for r in reports:
            name = r.get("scale_name", "")
            risk = r.get("risk_name", "")
            detail = ""
            if "subscales" in r:
                parts = [f"{v['label']}: {'⚠️阳性' if v['met'] else '✅阴性'}" for v in r["subscales"].values()]
                detail = "；".join(parts)
            elif "standard_score" in r:
                detail = f"标准分{r['standard_score']}"
            elif "often_count" in r:
                detail = f"总分{r['total']}，经常项{r['often_count']}"
            else:
                detail = f"总分{r.get('total', 0)}"
            scale_lines.append(f"- **{name}**：{risk}（{detail}）")

        scale_summary = "\n".join(scale_lines)

        prompt = f"""你是一位特殊教育IEP（个性化教育计划）专家。请为以下儿童生成《个性化教育计划建议书》初稿。

【儿童信息】
- 姓名：{child.get('name', '未知')}
- 年龄：{child.get('age', '?')}岁
- 性别：{child.get('gender', '未知')}
- 家长主诉：{ctx.get('concerns', '全面筛查')}

【评估结果】
{scale_summary}

【分班建议】
- 建议班型：{class_suggestion.get('class_type', '')}
- 支持策略：{class_suggestion.get('support', '')}

请生成结构化的Markdown格式IEP初稿，包含以下章节：

## 一、当前能力基线
基于评估结果，从社交沟通、认知学习、行为情绪、生活自理四个维度描述儿童当前能力水平。

## 二、学期目标（SMART原则）
列出3-5个具体、可衡量、可达成、相关、有时限的学期目标。例如：
- 社交目标：能在小组活动中主动与同伴互动，每周至少3次
- 学业目标：能在15分钟内独立完成数学作业，正确率达到80%
- 行为目标：能在课堂提醒下保持坐姿，连续专注时间达到20分钟

## 三、支持策略与调整
1. 课堂支持：座位安排、同伴支持、视觉提示等
2. 教学方法：任务分解、多感官教学、正向强化等
3. 环境调整：减少干扰、结构化日程、感统 breaks 等
4. 家校协作：家长配合事项、每日沟通方式

## 四、评估周期
建议何时进行中期评估和学期末评估，以及评估方式。

要求：
- 目标必须具体可量化，避免空泛描述
- 策略必须切合实际，能在普通班级或资源教室实施
- 语气专业、温暖、鼓励性
- 200-300字"""

        try:
            iep_text = call_llm_simple(prompt)
        except Exception as e:
            print(f"[WARN] IEP生成失败: {e}")
            iep_text = "IEP初稿生成失败，建议结合评估结果手动制定。"

        ctx["iep"] = {
            "generated": True,
            "content": iep_text,
            "risk_level": risk_level
        }
        print(f"[ASSESS] IEPNode: generated for risk={risk_level}")
        return "next", ctx["iep"]
    
class ReportNode(Node):
    async def exec(self, payload, ctx):
        child = ctx["child"]
        reports = ctx.get("all_reports", [])
        class_suggestion = ctx.get("class_suggestion", {})

        scale_lines = []
        for r in reports:
            name = r.get("scale_name", "")
            risk = r.get("risk_name", "")
            detail = ""
            if "subscales" in r:
                parts = [f"{v['label']}: {'⚠️阳性' if v['met'] else '✅阴性'}({v['count']}/{v['threshold']})" for v in r["subscales"].values()]
                detail = "；".join(parts)
            elif "standard_score" in r:
                detail = f"标准分{r['standard_score']}"
            elif "often_count" in r:
                detail = f"总分{r['total']}，经常项{r['often_count']}"
            else:
                detail = f"总分{r.get('total', 0)}"
            scale_lines.append(f"- **{name}**：{risk}（{detail}）")

        scale_summary = "\n".join(scale_lines)
        sugg = class_suggestion

        prompt = f"""请为以下特需儿童筛查结果写一段综合评估建议（300-400字）：

儿童：{child.get('name', '未知')}，{child.get('age', '?')}岁，主诉：{ctx.get('concerns', '全面筛查')}

【量表结果】
{scale_summary}

【分班建议】
{sugg.get('class_type', '')}
{sugg.get('reason', '')}
支持策略：{sugg.get('support', '')}

要求：像医生耐心讲解，给出就医、学校、家庭三方面具体行动，结尾强调不能替代医学诊断。"""

        try:
            advice_text = call_llm_simple(prompt)
        except Exception:
            advice_text = sugg.get("support", "建议根据结果采取相应措施。")

        risk_levels = [r.get("risk_level", "green") for r in reports]
        risk_priority = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
        max_risk = max(risk_levels, key=lambda x: risk_priority.get(x, 0))

                # IEP 章节（中高风险才显示）
        iep = ctx.get("iep")
        iep_section = ""
        if iep and iep.get("generated"):
            iep_section = f"\n\n## 七、个性化教育计划（IEP）建议书\n\n{iep['content']}"

        report_md = f"""# 📋 特需儿童入学能力评估报告

## 一、儿童基本信息
- **姓名**：{child.get('name', '未知')}
- **年龄**：{child.get('age', '?')} 岁
- **性别**：{child.get('gender', '未知')}
- **家长主诉**：{ctx.get('concerns', '全面筛查')}

## 二、量表筛查结果
{scale_summary}

## 三、风险分析与解读
- **综合风险等级**：{'🔴 高风险' if max_risk=='red' else '🟡 中风险' if max_risk=='yellow' else '🟢 低风险'}
- **核心发现**：{sugg.get('reason', '')}

## 四、分班建议（结构化匹配）
- **建议班型**：{sugg.get('class_type', '')}
- **安排方式**：{sugg.get('arrangement', '')}
- **依据**：根据爱华小学融合教育政策，{sugg.get('reason', '')}

## 五、支持策略与资源
{sugg.get('support', '')}

## 六、综合建议与行动指南
{advice_text}{iep_section}

---

⚠️ **免责声明**：本评估仅为筛查工具，不能替代专业医学诊断。建议结合临床医生的综合判断，制定最终教育方案。
"""

        full_report = {"child": child, "scales": reports, "class_suggestion": class_suggestion, "risk_level": max_risk, "risk_name": max_risk, "report_text": report_md}
        ctx["report"] = full_report
        ctx["status"] = "completed"
        print(f"[ASSESS] ReportNode: report_generated, risk={max_risk}")
        return "done", full_report


# ================== 聊天节点 ==================

class ChatNode(Node):
    async def exec(self, payload, ctx):
        messages = ctx["messages"]
        tools = ctx["tools"]
        system_prompt = ctx["system_prompt"]

        print(f"[LLM] ChatNode: sending {len(messages)} messages, tools_count={len(tools) if tools else 0}")
        assistant_message = call_llm(messages=messages, tools=tools, system_prompt=system_prompt)
        messages.append(assistant_message)

        has_tool = bool(assistant_message.get("tool_calls"))
        tool_names = [tc.get("function", {}).get("name", "") for tc in assistant_message.get("tool_calls", [])]
        print(f"[LLM] ChatNode: received assistant_message, has_tool={has_tool}, tools={tool_names}")

        if has_tool:
            for tc in assistant_message["tool_calls"]:
                func_name = tc.get("function", {}).get("name", "")
                if func_name == "search_knowledge_base":
                    print(f"[LLM] ChatNode -> action='rag'")
                    return "rag", assistant_message
                elif func_name == "start_assessment":
                    print(f"[LLM] ChatNode -> action='assessment'")
                    return "assessment", assistant_message
            print(f"[LLM] ChatNode -> action='tool_call'")
            return "tool_call", assistant_message

        print(f"[LLM] ChatNode -> action='output'")
        return "output", assistant_message


class ToolCallNode(Node):
    async def exec(self, payload, ctx):
        response = payload
        messages = ctx["messages"]
        store = ctx.get("assessment_store")

        tool_calls_to_execute = []
        assessment_call = None

        for tc in response.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except:
                    args = {}

            if name == "start_assessment":
                assessment_call = (tc, args)
            elif name == "search_knowledge_base":
                continue
            else:
                tool_calls_to_execute.append((tc, args))

        print(f"[TOOL] ToolCallNode: assessment_call={assessment_call is not None}, other_tools={len(tool_calls_to_execute)}")

        # 处理评估工具
        if assessment_call and store:
            tc, args = assessment_call
            tool_call_id = tc.get("id", "")
            
            try:
                sid = ctx.get("session_id", "")
                concerns = args.get("concerns", "")
                print(f"[TOOL] start_assessment: name={args.get('name')}, age={args.get('age')}, concerns={concerns}, session={sid}")

                asm_id = store.create("mchat_r", {
                    "name": args.get("name", "未知"),
                    "age": args.get("age", 0),
                    "gender": args.get("gender", "未知")
                }, concerns=concerns, session_id=sid)
                actx = store.get(asm_id)

                StartNode.init(actx, "mchat_r")
                actx["child"] = {"name": args.get("name", "未知"), "age": args.get("age", 0), "gender": args.get("gender", "未知")}
                actx["concerns"] = concerns

                router = ScaleRouterNode()
                await router._exec(None, actx)

                await self._pregenerate_questions(actx)

                qnode = QuestionNode()
                action, q_data = await qnode._exec(None, actx)

                store.update(asm_id, **actx)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({
                        "status": "assessment_created",
                        "assessment_id": asm_id,
                        "message": f"已为 {args.get('name')} 创建评估，请开始答题。",
                        "first_question": q_data,
                        "selected_scales": actx.get("selected_scales", []),
                        "llm_questions": actx.get("llm_questions", [])
                    }, ensure_ascii=False)
                })
                print(f"[TOOL] assessment_created: asm_id={asm_id}")
                
            except Exception as e:
                print(f"[ERROR] start_assessment failed: {e}")
                import traceback
                traceback.print_exc()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({
                        "status": "error",
                        "message": f"创建评估失败: {str(e)}"
                    }, ensure_ascii=False)
                })
            
            return "chat", None

        # 处理其他通用工具
        if not tool_calls_to_execute:
            return "chat", None

        # 执行工具，收集结果
        for tc, args in tool_calls_to_execute:
            name = tc.get("function", {}).get("name", "")
            tool_call_id = tc.get("id", "")
            
            # 防循环：检查是否最近刚失败过
            recent_failures = sum(
                1 for msg in messages[-10:] 
                if msg.get("role") == "tool" 
                and msg.get("tool_call_id") == tool_call_id
                and ("失败" in msg.get("content", "") or "Error" in msg.get("content", ""))
            )
            
            if recent_failures >= 2:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"【工具多次失败】{name} 已连续失败 {recent_failures} 次，请换用其他方案或向用户解释，不要再次调用此工具。"
                })
                continue

            output = await self._execute_one_tool(name, args)
            
            if output.startswith("Error:"):
                friendly = self._friendly_error(name, output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": friendly
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": output
                })

        return "chat", None

    def _friendly_error(self, tool_name, raw_error):
        """把技术错误翻译成 LLM 友好的描述"""
        error_map = {
            "bash": "命令执行失败",
            "read": "文件读取失败",
            "write": "文件写入失败",
            "edit": "文件修改失败",
            "grep": "搜索失败",
            "find": "查找失败",
        }
        category = error_map.get(tool_name, "操作失败")
        short_error = raw_error.replace("Error:", "").strip()[:100]
        return f"【{category}】{short_error}。请检查参数是否正确，或尝试其他方式。"

    async def _pregenerate_questions(self, ctx):
        """本地模板替换：零延迟，无需 LLM，支持人称替换 + 关键词举例"""
        child = ctx.get("child", {})
        child_name = child.get("name", "孩子")
        gender = child.get("gender", "")
        pronoun = "他" if gender == "男" else "她" if gender == "女" else "他/她"

        questions = ctx.get("questions", [])
        if not questions:
            ctx["llm_questions"] = []
            return

        examples = {
            "细节": "比如把 6 看成 9，漏写单位，或抄错行",
            "粗心": "比如把 6 看成 9，漏写单位，或抄错行",
            "专注": "比如上课上着上着就开始玩橡皮、看窗外",
            "听": "比如你跟她说'把书收起来准备吃饭'，她好像没听见，继续做自己的事",
            "遵循": "比如让她先写数学再写语文，她总是搞混顺序",
            "指示": "比如让她先写数学再写语文，她总是搞混顺序",
            "组织": "比如书包里东西乱塞，经常找不到作业本",
            "规划": "比如书包里东西乱塞，经常找不到作业本",
            "逃避": "比如一遇到要动脑筋的作业，就说'我不会'或'我不想做'",
            "动脑": "比如一遇到要动脑筋的作业，就说'我不会'或'我不想做'",
            "弄丢": "比如铅笔、橡皮、水杯经常不见了",
            "丢": "比如铅笔、橡皮、水杯经常不见了",
            "分心": "比如窗外有人走过，她马上转头去看，忘了刚才在做什么",
            "外在刺激": "比如窗外有人走过，她马上转头去看，忘了刚才在做什么",
            "忘东忘西": "比如早上提醒她带的东西，到学校发现还是忘了",
            "忘记": "比如早上提醒她带的东西，到学校发现还是忘了",
            "玩弄手脚": "比如吃饭、写作业时扭来扭去，或者找借口站起来走动",
            "坐不住": "比如吃饭、写作业时扭来扭去，或者找借口站起来走动",
            "离开座位": "比如在教室或电影院里，忍不住站起来走动",
            "乱跑": "比如在商场或教室里忍不住跑来跑去，难以自控",
            "爬高": "比如在商场或教室里忍不住跑来跑去，难以自控",
            "安静": "比如看电影或听故事时，手脚总是动个不停",
            "动个不停": "比如像装了小马达一样，一刻也闲不住",
            "马达": "比如像装了小马达一样，一刻也闲不住",
            "话多": "比如大人说话时她插嘴，或者一个人不停地说",
            "急着回答": "比如老师问题还没说完，她就急着喊答案",
            "插嘴": "比如老师问题还没说完，她就急着喊答案",
            "排队": "比如排队时推前面的人，或者一直问'还要多久'",
            "等待": "比如排队时推前面的人，或者一直问'还要多久'",
            "打断": "比如别的小朋友玩游戏，她非要过去插一脚",
            "干扰": "比如别的小朋友玩游戏，她非要过去插一脚",
            "发脾气": "比如因为小事突然大哭、摔东西，难以安抚",
            "争论": "比如明明是她错了，还要争辩'是他先动我的'",
            "顶嘴": "比如明明是她错了，还要争辩'是他先动我的'",
            "反抗": "比如让她收拾玩具，她偏不，甚至故意弄更乱",
            "拒绝": "比如让她收拾玩具，她偏不，甚至故意弄更乱",
            "惹恼": "比如故意去拍同学的头，或者把别人的东西藏起来",
            "怪罪": "比如作业没做完，她说'是妈妈没提醒我'",
            "敏感": "比如同学看她一眼，她就觉得别人在嘲笑她",
            "激怒": "比如同学看她一眼，她就觉得别人在嘲笑她",
            "生气": "比如心里一直记着谁惹了她，反复跟家长告状",
            "怨恨": "比如心里一直记着谁惹了她，反复跟家长告状",
            "报复": "比如故意把别人的作业本藏起来，因为对方没借她橡皮",
            "社交": "比如别的小朋友邀请她玩，她不知道怎么回应，就躲开了",
            "目光": "比如说话时眼睛看地上或旁边，不看你",
            "眼神": "比如说话时眼睛看地上或旁边，不看你",
            "模仿": "比如别的小朋友做手势游戏，她站在旁边不动",
            "名字": "比如你在身后叫她，她好像没听见，不回头",
            "尖叫": "比如想要东西时，不是指或说，而是直接大哭大叫",
            "哭闹": "比如想要东西时，不是指或说，而是直接大哭大叫",
            "刻板": "比如每天上学必须走同一条路，换了就哭闹",
            "强迫": "比如每天上学必须走同一条路，换了就哭闹",
            "痴迷": "比如对瓶盖、绳子、塑料袋着迷，玩几个小时",
            "旋转": "比如盯着风扇、车轮看很久，或者自己原地转圈",
            "重复": "比如无意义地拍手、摇晃身体，停不下来",
            "拍手": "比如无意义地拍手、摇晃身体，停不下来",
            "感觉": "比如只穿某一种材质的衣服，或者对特定声音特别敏感",
            "痛觉": "比如摔破膝盖也不哭，好像感觉不到疼",
            "痛": "比如摔破膝盖也不哭，好像感觉不到疼",
            "假装": "比如不会玩过家家，不会假装打电话或喂娃娃吃饭",
            "游戏": "比如别的小朋友玩捉迷藏，她站在旁边看，不参与",
            "小朋友": "比如别的小朋友玩捉迷藏，她站在旁边看，不参与",
            "感兴趣": "比如对小朋友不感兴趣，更喜欢自己一个人玩玩具",
            "危险": "比如喜欢爬柜子、沙发，不顾危险",
            "拥抱": "比如家人想抱抱她，她身体僵硬或躲开",
            "亲昵": "比如家人想抱抱她，她身体僵硬或躲开",
            "微笑": "比如你对她笑，她很少回以微笑",
            "笑容": "比如你对她笑，她很少回以微笑",
            "需要": "比如想要东西时，不是用手指或说，而是拉着你过去",
            "手势": "比如想要东西时，不是用手指或说，而是拉着你过去",
            "莫名其妙": "比如突然自己笑起来，不是因为看到了好笑的事",
            "笑": "比如突然自己笑起来，不是因为看到了好笑的事",
            "漠不关心": "比如对周围发生什么事都不好奇，像活在自己的世界里",
            "周围": "比如对周围发生什么事都不好奇，像活在自己的世界里",
        }

        llm_questions = []
        for q in questions:
            text = q["text"]
            text = text.replace("您的孩子", child_name)
            text = text.replace("您的", f"{child_name}的")
            text = text.replace("他/她", pronoun)
            text = text.replace("他", pronoun)
            text = text.replace("她", pronoun)

            added = False
            if "（" not in text:
                for keyword, example in examples.items():
                    if not added and keyword in text:
                        text = f"{text}（{example}）"
                        added = True

            llm_questions.append({
                "text": text,
                "options": [o["label"] for o in q.get("options", [])]
            })

        ctx["llm_questions"] = llm_questions
        print(f"[ASSESS] _pregenerate_questions: done, {len(llm_questions)} questions")

    async def _execute_one_tool(self, name, args):
        """执行单个工具，所有异常在此捕获，返回字符串"""
        try:
            if name == "bash":
                command = args.get("command", "")
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    return (stdout.decode('utf-8', errors='replace') + stderr.decode('utf-8', errors='replace')).strip()
                else:
                    return f"Error: 命令执行失败 (exit code {proc.returncode}): {stderr.decode('utf-8', errors='replace')[:200]}"

            elif name == "read":
                try:
                    path = args.get("path", "")
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    offset = args.get("offset", 0)
                    limit = args.get("limit", 50)
                    selected = lines[offset:offset + limit]
                    output = "".join(selected)
                    if len(lines) > offset + limit:
                        output += f"\n...（还有 {len(lines) - offset - limit} 行未显示）"
                    return output
                except Exception as e:
                    return f"Error: {e}"

            elif name == "write":
                try:
                    path = args.get("path", "")
                    content = args.get("content", "")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    return f"文件已写入：{path}（{len(content)} 字符）"
                except Exception as e:
                    return f"Error: {e}"

            elif name == "edit":
                try:
                    path = args.get("path", "")
                    old_s = args.get("old_string", "")
                    new_s = args.get("new_string", "")
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if old_s not in content:
                        return f"Error: 未找到 '{old_s}'"
                    new_content = content.replace(old_s, new_s, 1)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    return f"文件已修改：{path}"
                except Exception as e:
                    return f"Error: {e}"

            elif name == "grep":
                try:
                    path = args.get("path", "")
                    pattern = args.get("pattern", "")
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    matches = []
                    for i, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            matches.append(f"第{i}行: {line.strip()}")
                    return "\n".join(matches[:20]) if matches else f"未找到匹配 '{pattern}'"
                except Exception as e:
                    return f"Error: {e}"

            elif name == "find":
                try:
                    path = args.get("path", ".")
                    pattern = args.get("pattern", "*")
                    matches = []
                    for root, dirs, files in os.walk(path):
                        for filename in files:
                            if fnmatch.fnmatch(filename, pattern):
                                matches.append(os.path.join(root, filename))
                    return "\n".join(matches[:20]) if matches else f"未找到匹配 '{pattern}'"
                except Exception as e:
                    return f"Error: {e}"

            else:
                return f"Error: unknown tool '{name}'"

        except Exception as e:
            return f"Error: 工具执行异常 - {str(e)[:200]}"

class KnowledgeNode(Node):
    async def exec(self, payload, ctx):
        response = payload
        messages = ctx["messages"]
        collection = ctx.get("chroma_collection")

        query = ""
        tool_call_id = ""
        for tc in response.get("tool_calls", []):
            func = tc.get("function", {})
            if func.get("name") == "search_knowledge_base":
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}
                query = args.get("query", "")
                tool_call_id = tc.get("id", "")
                break

        if not collection:
            context = "知识库未初始化"
        else:
            try:
                results = collection.query(query_texts=[query], n_results=3)
                docs = results.get("documents", [[]])[0]
                context = "\n\n".join(docs) if docs else "未找到相关内容"
            except Exception as e:
                context = f"检索出错: {e}"

        messages.append({"role": "tool", "tool_call_id": tool_call_id or "kb_fallback", "content": f"知识库检索结果：\n{context}"})
        print(f"[RAG] KnowledgeNode: query='{query}', results={len(docs) if 'docs' in dir() else 0}")
        return "chat", None


class OutputNode(Node):
    async def exec(self, payload, ctx):
        return "default", payload