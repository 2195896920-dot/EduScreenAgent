# core/assessment_store.py
import json
import os
import uuid

ASSESSMENTS_FILE = "assessments.json"

class AssessmentStore:
    def __init__(self):
        self.assessments = {}
        self._load()

    def _load(self):
        if os.path.exists(ASSESSMENTS_FILE):
            try:
                with open(ASSESSMENTS_FILE, "r", encoding="utf-8") as f:
                    self.assessments = json.load(f)
            except Exception as e:
                print(f"[WARN] 加载评估数据失败: {e}")
                self.assessments = {}

    def _save(self):
        try:
            with open(ASSESSMENTS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.assessments, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 保存评估数据失败: {e}")

    def create(self, scale_name, child_info, concerns="", session_id=""):
        asm_id = f"asm_{uuid.uuid4().hex[:8]}"
        self.assessments[asm_id] = {
            "scale_name": scale_name,
            "child": child_info,
            "concerns": concerns,
            "session_id": session_id,
            "answers": [],
            "current_qid": 0,
            "status": "created",
            "report": None,
            "current_question": None,
            "total": 0,
            "scale": None,
            "questions": None,
            "scores": None,
            "risk_level": None,
            "risk_name": None,
            "selected_scales": [],
            "current_scale_index": 0,
            "current_scale_name": "",
            "all_reports": [],
            "class_suggestion": None,
            "llm_questions": []
        }
        self._save()
        return asm_id

    def get(self, asm_id):
        return self.assessments.get(asm_id)

    def update(self, asm_id, **kwargs):
        if asm_id in self.assessments:
            self.assessments[asm_id].update(kwargs)
            self._save()

    def all(self):
        return self.assessments