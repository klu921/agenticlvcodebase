#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import shutil
import subprocess
import ffmpeg
from pathlib import Path


def find_video_file(video_dir: Path) -> Path:
    # Prefer mp4, then common formats
    exts = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
    candidates = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not candidates:
        raise FileNotFoundError(f"No video file found in {video_dir} (looked for: {', '.join(exts)})")
    if len(candidates) > 1:
        # Choose the largest file as likely the main video
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def extract_frames_ffmpeg(video_file: Path, frames_dir: Path, every_seconds: int = 2) -> None:
    """Extract frames using ffmpeg binary if available."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found in PATH")
    cmd = [
        "ffmpeg",
        "-i",
        str(video_file),
        "-vf",
        f"fps=1/{every_seconds}",
        "-q:v",
        "2",
        str(frames_dir / "frame_%04d.jpg"),
        "-y",
    ]
    subprocess.run(cmd, check=True)


def extract_frames_opencv(video_file: Path, frames_dir: Path, every_seconds: int = 2) -> None:
    """Fallback extractor using OpenCV if ffmpeg is unavailable."""
    try:
        import cv2
    except Exception as e:
        raise FileNotFoundError("Neither ffmpeg (binary) nor OpenCV is available for frame extraction") from e

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 30.0  # default fallback
    step = int(round(fps * every_seconds))
    frame_idx = 0
    saved_idx = 0
    ok, frame = cap.read()
    while ok:
        if frame_idx % max(step, 1) == 0:
            saved_idx += 1
            out_path = frames_dir / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
        frame_idx += 1
        ok, frame = cap.read()
    cap.release()


async def generate_captions(frames_dir: Path, captions_dir: Path) -> None:
    from caption_frames_os import caption_frame_with_os

    captions_dir.mkdir(parents=True, exist_ok=True)
    out_file = captions_dir / "frame_captions.json"
    await caption_frame_with_os(frames_dir=str(frames_dir), output_file=str(out_file), max_concurrent=10)


def sort_captions(captions_dir: Path) -> Path:
    src = captions_dir / "frame_captions.json"
    dst = captions_dir / "frame_captions_sorted.json"
    if not src.exists():
        raise FileNotFoundError(f"Missing captions: {src}")
    with open(src, "r") as f:
        arr = json.load(f)
    # Entries are strings like "frames/frame_XXXX seconds: caption"
    try:
        sorted_arr = sorted(arr, key=lambda x: str(x).split(" seconds:")[0])
    except Exception:
        sorted_arr = arr
    with open(dst, "w") as f:
        json.dump(sorted_arr, f, indent=2)
    return dst


async def embed_captions(sorted_path: Path) -> Path:
    from embed_frame_captions import embed_one

    out_path = sorted_path.parent / "frame_captions_sorted_embeddings.jsonl"
    await embed_one(str(sorted_path), str(out_path))
    return out_path


async def generate_logs(frames_dir: Path, captions_dir: Path) -> None:
    from caption_frames_os import create_logs, CES_log_prompt, global_summary_prompt

    sorted_path = captions_dir / "frame_captions_sorted.json"
    ces_file = captions_dir / "CES_logs.txt"
    gs_file = captions_dir / "global_summary.txt"
    # CES logs
    await create_logs(
        captions_dir=str(sorted_path),
        output_file=str(ces_file),
        prompt_fct=CES_log_prompt,
        frames_dir=str(frames_dir),
    )
    # Global summary
    await create_logs(
        captions_dir=str(sorted_path),
        output_file=str(gs_file),
        prompt_fct=global_summary_prompt,
        frames_dir=str(frames_dir),
    )


async def answer_question(vid_dir: Path, video_id: str, uid: str, question: str, llm_model: str, vlm_model: str):
    from one_question import run_one_question

    return await run_one_question(
        vid_dir=str(vid_dir),
        video_id=video_id,
        uid=uid,
        question=question,
        llm_model=llm_model,
        vlm_model=vlm_model,
    )


async def main():
    parser = argparse.ArgumentParser(description="Run end-to-end demo for a single video directory")
    parser.add_argument("video_dir", help="Path to a directory containing exactly one video file")
    parser.add_argument("question", help="Question to ask about the video")
    parser.add_argument("--uid", default="q1")
    parser.add_argument("--llm_model", default="openai/gpt-oss-120b")
    parser.add_argument("--vlm_model", default="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
    args = parser.parse_args()

    # Normalize paths
    video_dir = Path(args.video_dir).resolve()
    env_path = Path('env.json')
    if not env_path.exists():
        raise FileNotFoundError(f"Env.json not found. Please create it in this folder..")
    
    with open(env_path, "r") as f:
        env_data = json.load(f)
    os.environ['TOGETHER_API_KEY'] = env_data["together_key"]
    os.environ['GEMINI_API_KEY'] = env_data["gemini_key"]
    os.environ['OPENAI_API_KEY'] = env_data["openai_key"]
    
    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Invalid video_dir: {video_dir}")

    # Repo root = LVBench folder (demo/ is under it)
    repo_root = Path(__file__).resolve().parents[1]

    # Prepare structure under the provided video_dir
    video_id = video_dir.name
    frames_dir = video_dir / "frames"
    captions_dir = video_dir / "captions"
    os.makedirs(captions_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # 1) Extract frames every 2 seconds
    video_file = find_video_file(video_dir)
    # Try ffmpeg first; fallback to OpenCV if not installed
    try:
        extract_frames_ffmpeg(video_file, frames_dir, every_seconds=2)
    except FileNotFoundError:
        extract_frames_opencv(video_file, frames_dir, every_seconds=2)

    # 2) Caption frames
    await generate_captions(frames_dir, captions_dir)

    # 3) Sort captions and embed
    sorted_path = sort_captions(captions_dir)
    embedded_captions_path = captions_dir / "frame_captions_sorted_embeddings.jsonl"
    if embedded_captions_path.exists():
        print("Embedded captions already exist. Skipping embedding.")
    else:
        print("Embedding captions...")
        await embed_captions(sorted_path)

    # 4) Generate logs required by the downstream pipeline
    await generate_logs(frames_dir, captions_dir)

    # 5) Ask the question via the existing pipeline
    result = await answer_question(
        vid_dir=video_dir.parent,
        video_id=video_id,
        uid=args.uid,
        question=args.question,
        llm_model=args.llm_model,
        vlm_model=args.vlm_model,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


