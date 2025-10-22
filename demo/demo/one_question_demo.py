#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from typing import Dict, Any

from os_model_demo import Pipeline as OSPipelineDemo, query_model_iterative_with_retry as os_query_with_retry_demo
from critic_model_os import CriticPipeline, critic_assess
from critic_response import Pipeline as CriticRespPipeline, query_model_iterative_with_retry as critic_resp_query_with_retry, create_enhanced_prompt


async def run_one_question_demo(
    vid_dir: str,
    question: str,
    llm_model: str = "openai/gpt-oss-120b",
    vlm_model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
) -> Dict[str, Any]:
    # Determine if vid_dir is the video folder or its parent
    if os.path.isdir(os.path.join(vid_dir, "frames")) and os.path.isdir(os.path.join(vid_dir, "captions")):
        vid_root = os.path.dirname(vid_dir)
        video_id = os.path.basename(vid_dir)
    else:
        subs = [d for d in os.listdir(vid_dir) if os.path.isdir(os.path.join(vid_dir, d))]
        if len(subs) != 1:
            raise RuntimeError("vid_dir should be the video folder (with frames/captions) or a parent with exactly one video subdir")
        video_id = subs[0]
        vid_root = vid_dir

    vid_path = f"{vid_root}/{video_id}"

    # Required context
    global_summary_path = f"{vid_path}/captions/global_summary.txt"
    ces_logs_path = f"{vid_path}/captions/CES_logs.txt"
    if not os.path.exists(global_summary_path):
        raise FileNotFoundError(f"Missing global summary: {global_summary_path}")
    if not os.path.exists(ces_logs_path):
        raise FileNotFoundError(f"Missing CES logs: {ces_logs_path}")

    # 1) OS model answer (no multiple choice)
    os_model = OSPipelineDemo(llm_model, vlm_model)
    answer = await os_query_with_retry_demo(os_model, question, vid_path)
    if not answer or not isinstance(answer, dict):
        raise RuntimeError("OS demo model did not return a valid answer dict")

    # 2) Critic assessment (re-uses existing critic pipeline but free-form answer)
    critic = CriticPipeline(llm_model, vlm_model)
    assessment = await critic_assess(
        critic,
        question,
        "demo",  # uid placeholder
        answer.get("answer"),
        answer.get("reasoning", ""),
        answer.get("evidence_frame_numbers", []),
        vid_root,
        video_id,
    )

    # 3) Critic response re-evaluation
    enhanced_question = create_enhanced_prompt(assessment)
    critic_resp_model = CriticRespPipeline(llm_model, vlm_model)
    re_eval = await critic_resp_query_with_retry(
        critic_resp_model,
        enhanced_question,
        "demo",
        vid_path,
    )

    return {
        "os_answer": answer,
        "critic_assessment": assessment,
        "critic_re_evaluation": re_eval,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run demo OS -> Critic -> Critic-Response for one free-form question on one video",
    )
    parser.add_argument("vid_dir", help="Video directory or its parent containing the single video folder")
    parser.add_argument("question", help="Free-form question text to answer")
    parser.add_argument("--llm_model", default="openai/gpt-oss-120b")
    parser.add_argument("--vlm_model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            run_one_question_demo(
                args.vid_dir,
                args.question,
                args.llm_model,
                args.vlm_model,
            )
        )
    finally:
        loop.close()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


