### Demo: Run the end-to-end video QA pipeline on one video

This demo extracts frames from a single video, captions them, embeds captions, generates logs, and answers a free-form question about the video.

### Prerequisites
- Python 3.10+
- One of:
  - ffmpeg installed in PATH
  - OR Python OpenCV (`opencv-python`) for frame extraction fallback
- API keys in `env.json` (OpenAI required; Together optional)

Install Python deps:
```bash
cd demo
pip install -r demo/requirements.txt
```

Install ffmpeg (recommended):
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Or rely on fallback: `pip install opencv-python`

### Prepare environment
Create an env folder containing `env.json` like:
```json
{
  "openai_key": "sk-...",
  "together_key": "tg-..."
}
```

### Video folder structure
Place exactly one video file inside a folder you make, e.g. `demo_video/`:
```
demo_video/
  your_video.mp4   # any common format is fine, mp4 is preferred.
```
The demo will create `frames/` and `captions/` inside this folder.

### End-to-end: extract → captions → embeddings → logs → answer
Runs the full pipeline and prints the JSON answer.
```bash
python LVBench/demo/run_demo_one_video.py /abs/path/to/demo_video /abs/path/to/env_folder "Your free-form question here" --uid q1
```
- Outputs are written under `/abs/path/to/demo_video/`.

### Just Q&A (if frames/captions already exist)
If you’ve already run extraction/captioning/embedding/logs, run the localized no-multiple-choice pipeline:
```bash
python LVBench/demo/one_question_demo.py /abs/path/to/demo_video "Your free-form question here"
```

### Outputs
Under your video folder:
- `frames/` — extracted frames every 2 seconds (or via fallback)
- `captions/frame_captions.json` — per-frame captions
- `captions/frame_captions_sorted.json` — sorted captions
- `captions/frame_captions_sorted_embeddings.jsonl` — caption embeddings
- `captions/CES_logs.txt` — character/event/scene logs
- `captions/global_summary.txt` — global video summary
- `<video_id>_answers.json`, `<video_id>_critic_assessment.json` — pipeline results

### Troubleshooting
- ffmpeg not found: install ffmpeg or `pip install opencv-python` (fallback is automatic)
- Missing API key: ensure `env.json` is present and has a valid `togetherai_key`
- Rate limits/timeouts: re-run; the demo handles retries in several steps

### Notes
- The demo uses localized, free-form prompts (no multiple-choice) under `LVBench/demo/`.
- For reproducibility, prefer absolute paths in commands.


See our blog and demo here! 

https://klu921.github.io/agentic_lv_blog/
https://klu921.github.io/agentic_lv_demo/
