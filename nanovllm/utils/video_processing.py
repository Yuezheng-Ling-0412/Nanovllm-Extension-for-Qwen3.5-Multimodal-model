import json
import os
import shutil
import subprocess
from typing import Literal

import numpy as np
from PIL import Image


def _parse_frame_rate(rate: str | None) -> float | None:
    if not rate:
        return None
    num, _, den = rate.partition("/")
    try:
        num, den = float(num), float(den or 1)
    except ValueError:
        return None
    if num <= 0 or den <= 0:
        return None
    return num / den


def probe_video(video_path: str) -> dict:

    if shutil.which("ffprobe") is None:
        raise RuntimeError("未找到 ffprobe，请先安装 ffmpeg（apt install ffmpeg）")

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,width,height",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ValueError(f"无法解析视频 {video_path}: {proc.stderr.strip()}")

    info = json.loads(proc.stdout or "{}")
    streams = info.get("streams") or []
    if not streams:
        raise ValueError(f"{video_path} 中找不到视频流（可能不是视频文件）")
    stream = streams[0]

    fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(stream.get("r_frame_rate"))
    duration = stream.get("duration") or (info.get("format") or {}).get("duration")
    try:
        duration = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration = None
    nb_frames = stream.get("nb_frames")
    try:
        nb_frames = int(nb_frames) if nb_frames is not None else None
    except (TypeError, ValueError):
        nb_frames = None

    if fps is None and nb_frames and duration:
        fps = nb_frames / duration
    if fps is None:
        raise ValueError(f"无法从 {video_path} 读取帧率，无法计算时间步")

    return {
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": nb_frames,
        "duration": duration,
    }


def _sample_frames_decord(video_path: str, frame_interval: int) -> tuple[np.ndarray, float]:
    import decord

    reader = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
    indices = list(range(0, len(reader), frame_interval))
    if not indices:
        raise ValueError(f"没有从 {video_path} 解析到任何视频帧")
    frames = reader.get_batch(indices).asnumpy()
    return frames, float(reader.get_avg_fps())


def _sample_frames_ffmpeg(video_path: str, frame_interval: int) -> tuple[np.ndarray, float]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未找到 ffmpeg，请先安装（apt install ffmpeg）或 pip install decord")

    meta = probe_video(video_path)
    width, height = meta["width"], meta["height"]

    base_cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-i", video_path,
        "-vf", f"select='not(mod(n\\,{frame_interval}))'",
    ]
    tail_cmd = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    stdout, stderr = None, ""
    for out_fps_mode in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        proc = subprocess.run(base_cmd + out_fps_mode + tail_cmd, capture_output=True)
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        if proc.returncode == 0:
            stdout = proc.stdout
            break
        if "unrecognized option" not in stderr.lower():
            raise RuntimeError(f"ffmpeg 解码 {video_path} 失败: {stderr.strip()}")
    if stdout is None:
        raise RuntimeError(f"ffmpeg 解码 {video_path} 失败: {stderr.strip()}")

    frame_size = width * height * 3
    buffer = np.frombuffer(stdout, dtype=np.uint8)
    num_frames = buffer.size // frame_size
    if num_frames == 0:
        raise ValueError(f"没有从 {video_path} 解析到任何视频帧")
    frames = buffer[: num_frames * frame_size].reshape(num_frames, height, width, 3).copy()
    return frames, meta["fps"]


def sample_video_frames(
    video_path: str,
    frame_interval: int = 12,
    return_format: Literal["pil", "numpy"] = "pil",
) -> tuple[list, list[float]]:
    """按固定帧间隔对视频抽帧，帧数为奇数时复制最后一帧凑成偶数。

    Args:
        video_path: 视频文件路径
        frame_interval: 抽帧间隔（每隔多少帧取一帧），默认 12
        return_format: 返回帧的格式，"pil" 返回 PIL.Image（RGB），"numpy" 返回 (H, W, 3) 的 uint8 数组

    Returns:
        (frames, timesteps):
            frames: 抽帧后的帧列表，长度为偶数（奇数时复制了最后一帧）
            timesteps: 与 frames 一一对应的时间步列表，单位为秒
                       （复制出来的那帧沿用被复制帧的时间步）
    """
    if frame_interval < 1:
        raise ValueError(f"frame_interval 必须 >= 1，当前为 {frame_interval}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    try:
        frames, fps = _sample_frames_decord(video_path, frame_interval)
    except ImportError:
        frames, fps = _sample_frames_ffmpeg(video_path, frame_interval)

    timesteps = [i * frame_interval / fps for i in range(len(frames))]

    if len(frames) % 2 == 1:
        frames = np.concatenate([frames, frames[-1:]], axis=0)
        timesteps.append(timesteps[-1])

    if return_format == "numpy":
        return [f for f in frames], timesteps
    if return_format != "pil":
        raise ValueError(f"不支持的 return_format: {return_format}")
    return [Image.fromarray(f) for f in frames], timesteps
