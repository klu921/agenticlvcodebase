import json
import os
import asyncio
from typing import Dict, Any

from model_example_query import query_llm_async
from prompts_demo import (
    initial_prompt_no_mc,
    followup_prompt_no_mc,
    finish_prompt_no_mc,
)
from model_example_query import query_vlm
from search_frame_captions import search_captions


class Pipeline:
    def __init__(self, llm_model_name: str, vlm_model_name: str):
        self.llm_model = llm_model_name
        self.vlm_model = vlm_model_name
        self.messages = []

    async def llm(self, prompt: str) -> str:
        return await query_llm_async(self.llm_model, prompt)

    async def vlm_query(self, image_paths, prompt):
        return await query_vlm(self.vlm_model, image_paths, prompt)


async def query_model_iterative_with_retry(model: Pipeline, question: str, vid_path: str, max_retries: int = 15) -> Dict[str, Any]:
    prompt = initial_prompt_no_mc(question)
    for i in range(max_retries):
        response = await model.llm(prompt)
        if not response:
            continue
        model.messages.append({"role": "assistant", "content": response})

        # Try to parse as JSON
        try:
            parsed = None
            if "```json" in response:
                parsed = json.loads(response.split("```json")[1].split("```")[0].strip())
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                parsed = json.loads(response[start:end])
            else:
                parsed = json.loads(response.strip())
        except Exception:
            parsed = None

        if not parsed or not isinstance(parsed, dict):
            continue

        tool = parsed.get("tool")
        if tool == "FINAL_ANSWER":
            return {
                "question": question,
                "answer": parsed.get("answer", ""),
                "reasoning": parsed.get("reasoning", ""),
                "evidence_frame_numbers": parsed.get("frames", []),
            }
        elif tool == "VLM_QUERY":
            frames = parsed.get("frames", [])
            vlm_prompt = parsed.get("prompt", "")
            new_frames = [f"{vid_path}/" + f for f in frames]
            vlm_resp = await model.vlm_query(new_frames, vlm_prompt)
            model.messages.append({"role": "vlm response", "content": vlm_resp})
        elif tool == "CAPTION_SEARCH":
            search_query = parsed.get("prompt") or parsed.get("input")
            if isinstance(search_query, list):
                search_query = search_query[0]
            if search_query:
                retrieved_info = await search_captions(vid_path, "demo", search_query, f"{vid_path}/captions/frame_captions_sorted_embeddings.jsonl", 30)
                if isinstance(retrieved_info, list):
                    retrieved_info = json.dumps(retrieved_info, indent=2)
                model.messages.append({"role": "caption search results", "content": retrieved_info})
        else:
            # Unrecognized tool; continue iteration
            pass

        # Build next prompt
        prompt = followup_prompt_no_mc(model.messages, question)

    # Fallback finalization
    final_prompt = finish_prompt_no_mc(model.messages)
    final_answer = await model.llm(final_prompt)
    try:
        if isinstance(final_answer, str):
            if "```json" in final_answer:
                json_str = final_answer.split("```json")[1].split("```")[0].strip()
                parsed_final = json.loads(json_str)
            elif "{" in final_answer and "}" in final_answer:
                start = final_answer.find("{")
                end = final_answer.rfind("}") + 1
                parsed_final = json.loads(final_answer[start:end])
            else:
                return {
                    "question": question,
                    "answer": final_answer,
                    "reasoning": "Final iteration response",
                    "evidence_frame_numbers": [],
                }
            return {
                "question": question,
                "answer": parsed_final.get("answer", ""),
                "reasoning": parsed_final.get("reasoning", ""),
                "evidence_frame_numbers": parsed_final.get("frames", []),
            }
    except Exception:
        pass

    return {
        "question": question,
        "answer": "ERROR",
        "reasoning": "Failed to produce a final answer",
        "evidence_frame_numbers": [],
    }


