"""GEPA adapter for multimodal campaign-relevance classification via OpenAI Responses API."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ingest import _extract_openai_text  # noqa: E402

from gepa.image import Image

_REASONING_PREFIXES = ("gpt-5", "o3", "o4")
_TEMPLATE_PATH = ROOT / "config" / "request_template.json"
# Cap frames sent to the reflection VLM per example (strategy 1).
MAX_REFLECTION_IMAGES = 4


class CampaignDataInst(TypedDict):
    input: Dict[str, Any]
    answer: str
    additional_context: Dict[str, str]


class CampaignTrajectory(TypedDict):
    data: CampaignDataInst
    full_assistant_response: str
    feedback: str


class CampaignRolloutOutput(TypedDict):
    full_assistant_response: str


def _load_text_template() -> Dict[str, Any]:
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["text"]


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def _build_response_input(system_prompt: str, mm_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for msg in mm_input.get("raw_user_messages") or []:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "input_text":
                    blocks.append({"type": "input_text", "text": str(block.get("text") or "")})
                elif block.get("type") == "input_image":
                    blocks.append({"type": "input_image", "image_url": str(block.get("image_url") or "")})
            if blocks:
                messages.append({"role": role, "content": blocks})
    if len(messages) == 1:
        text = "\n".join(mm_input.get("text_parts") or [])
        messages.append({"role": "user", "content": text})
    return messages


def _parse_prediction(response_text: str) -> bool | None:
    try:
        payload = json.loads(response_text or "{}")
        if isinstance(payload, dict) and "campaign_relevant" in payload:
            return bool(payload["campaign_relevant"])
    except json.JSONDecodeError:
        pass
    upper = (response_text or "").upper()
    if "TRUE" in upper and "FALSE" not in upper:
        return True
    if "FALSE" in upper and "TRUE" not in upper:
        return False
    return None


class CampaignRelevanceEvaluator:
    def __call__(self, data: CampaignDataInst, response: str) -> tuple[float, str]:
        gold = data["answer"] == "TRUE"
        pred = _parse_prediction(response)
        if pred is None:
            return 0.0, (
                f"Could not parse prediction from model output. Expected answer={data['answer']}. "
                f"Output snippet: {response[:500]}"
            )
        score = 1.0 if pred == gold else 0.0
        if score == 1.0:
            feedback = f"Correct. Expected {data['answer']} and model predicted {pred}."
        else:
            ctx = ", ".join(f"{k}={v}" for k, v in data["additional_context"].items())
            feedback = (
                f"Incorrect. Expected {data['answer']} but model predicted {pred}. "
                f"Context: {ctx}. Review captions, mentions, hashtags, and images carefully."
            )
        return score, feedback


class CampaignRelevanceAdapter:
    # GEPA checks this attribute directly; None => use reflection_lm.
    propose_new_texts = None

    def __init__(self, model: str, max_workers: int = 8):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.evaluator = CampaignRelevanceEvaluator()
        self.text_template = _load_text_template()
        self.max_workers = max_workers

    def _call_model(self, system_prompt: str, data: CampaignDataInst) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": _build_response_input(system_prompt, data["input"]),
            "text": self.text_template,
        }
        if _is_reasoning_model(self.model):
            kwargs["reasoning"] = {"effort": "low"}
        response = self.client.responses.create(**kwargs)
        body = response.model_dump()
        text = _extract_openai_text(body) or ""
        return text.strip()

    def evaluate(self, batch: List[CampaignDataInst], candidate: Dict[str, str], capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        system_prompt = candidate.get("system_prompt") or next(iter(candidate.values()))
        outputs: List[CampaignRolloutOutput] = []
        scores: List[float] = []
        trajectories: List[CampaignTrajectory] | None = [] if capture_traces else None

        def _run_row(row: CampaignDataInst) -> tuple[str, float, str]:
            try:
                response_text = self._call_model(system_prompt, row)
            except Exception as exc:
                response_text = json.dumps(
                    {"campaign_relevant": False, "relevancy_reasoning": f"error: {exc}"}
                )
            score, feedback = self.evaluator(row, response_text)
            return response_text, score, feedback

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(_run_row, batch))

        for row, (response_text, score, feedback) in zip(batch, results, strict=True):
            outputs.append({"full_assistant_response": response_text})
            scores.append(score)
            if trajectories is not None:
                trajectories.append(
                    {
                        "data": row,
                        "full_assistant_response": response_text,
                        "feedback": feedback,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: Dict[str, str],
        eval_batch,
        components_to_update: List[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        comp = components_to_update[0]
        trajectories = eval_batch.trajectories
        if trajectories is None:
            raise RuntimeError("Trajectories are required for reflection")
        items = []
        for traj in trajectories:
            mm = traj["data"]["input"]
            text_preview = "\n".join(mm.get("text_parts") or [])[:4000]
            image_urls = (mm.get("image_urls") or [])[:MAX_REFLECTION_IMAGES]
            frames = [Image(url=str(url)) for url in image_urls if url]
            record: Dict[str, Any] = {
                "Inputs": text_preview,
                "Generated Outputs": traj["full_assistant_response"],
                "Feedback": traj["feedback"],
            }
            if frames:
                record["Frames"] = frames
            items.append(record)
        if not items:
            raise RuntimeError("No reflective records built")
        return {comp: items}
