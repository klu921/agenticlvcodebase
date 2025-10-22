from typing import Any


def initial_prompt_no_mc(question: str) -> str:
    return f"""
You are a reasoning model whose goal is to answer questions about a video. You cannot see the video, but you can use tools to retrieve information about the video.

You will be given a single free-form question (no multiple-choice).

Question: {question}

Given the above question, think step by step about what visual evidence is required. Then choose EXACTLY ONE of the following actions:
  1) CAPTION_SEARCH: Propose a short phrase to search the frame captions that would likely retrieve relevant frames.
  2) VLM_QUERY: Propose a set of frames to query with a visual-language model and a concise verification prompt.
  3) FINAL_ANSWER: If you ALREADY have sufficient evidence, provide a final answer and list the supporting frames.

Return ONLY valid JSON in one of the following formats:
For CAPTION_SEARCH:
{
  "tool": "CAPTION_SEARCH",
  "prompt": "short search phrase"
}

For VLM_QUERY:
{
  "tool": "VLM_QUERY",
  "frames": ["frames/frame_0123.jpg", "frames/frame_0456.jpg"],
  "prompt": "what to verify in these frames"
}

For FINAL_ANSWER:
{
  "tool": "FINAL_ANSWER",
  "frames": ["frames/frame_0123.jpg", ...],
  "answer": "free-form answer text",
  "reasoning": "brief reasoning using evidence"
}
"""


def followup_prompt_no_mc(json_output: Any, question: str) -> str:
    return f"""
Here is the result of your previous step:
{json_output}

Original question: {question}

Choose EXACTLY ONE next action, preferring to gather enough evidence before concluding:
  1) CAPTION_SEARCH: propose a short search phrase to retrieve better frames
  2) VLM_QUERY: propose frames and a concise verification prompt
  3) FINAL_ANSWER: when you have sufficient evidence

Return ONLY one of these JSON schemas (same as before).
"""


def finish_prompt_no_mc(scratchpad: Any) -> str:
    return f"""
Given all the retrieved information and VLM results below, provide a final free-form answer and list the supporting frames.

Scratchpad:
{scratchpad}

Return ONLY valid JSON:
{{
  "answer": "free-form answer text",
  "frames": ["frames/frame_0123.jpg", ...],
  "reasoning": "brief reasoning using evidence"
}}
IMPORTANT:
- Use only frames you actually referenced or that directly support the answer
- Frame paths must include "frames/" and end with ".jpg"
"""


