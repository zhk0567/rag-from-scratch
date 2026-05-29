"""视频帧抽取 + 视觉描述（可选 opencv）。"""

import tempfile
from pathlib import Path

import config
from logger import get_logger
from multimodal import describe_image_bytes

log = get_logger()

VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mov", ".mkv"}


def _extract_frames(video_path: Path) -> list[tuple[int, bytes]]:
    try:
        import cv2
    except ImportError as e:
        raise ImportError("视频处理需要: pip install opencv-python-headless") from e

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path.name}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(config.VIDEO_FRAME_INTERVAL_SEC, 1)
    frame_step = int(fps * interval)
    if frame_step < 1:
        frame_step = 1

    frames: list[tuple[int, bytes]] = []
    idx = 0
    captured = 0

    while captured < config.VIDEO_MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append((captured + 1, buf.tobytes()))
                captured += 1
        idx += 1

    cap.release()
    return frames


def read_video_as_text(path: Path) -> str:
    """按间隔抽帧，视觉描述后合并为文档文本。"""
    if not config.USE_MULTIMODAL:
        log.warning("跳过视频 %s（需 USE_MULTIMODAL=true）", path.name)
        return ""

    frames = _extract_frames(path)
    if not frames:
        return ""

    parts: list[str] = []
    with tempfile.TemporaryDirectory(prefix="rag_vid_") as tmpdir:
        tmp = Path(tmpdir)
        for seq, (frame_no, data) in enumerate(frames, start=1):
            img_path = tmp / f"frame_{frame_no}.jpg"
            img_path.write_bytes(data)
            desc = describe_image_bytes(data, label=f"{path.name}#f{frame_no}")
            if desc:
                t_sec = (frame_no - 1) * config.VIDEO_FRAME_INTERVAL_SEC
                parts.append(f"[视频 {path.name} · 约 {t_sec}s · 帧 {seq}]\n{desc}")

    if not parts:
        return ""

    return f"# 视频文档: {path.name}\n\n" + "\n\n".join(parts)
