#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import subprocess
import json
import hashlib
from loguru import logger
from typing import Dict, List, Optional
from pathlib import Path

from app.utils import ffmpeg_utils

def check_hardware_acceleration() -> Optional[str]:
    """
    检查系统支持的硬件加速选项

    Returns:
        Optional[str]: 硬件加速参数，如果不支持则返回None
    """
    # 使用集中式硬件加速检测
    return ffmpeg_utils.get_ffmpeg_hwaccel_type()

def generate_video_tts_srt(
        video_origin_path: str,
        tts_path: str,
        output_dir: Optional[str] = None,
        task_id: Optional[str] = None
) -> str:
    """
    根据时间戳裁剪视频

    Args:
        video_origin_path: 原始视频的路径
        tts_result: 包含时间戳和持续时间信息的列表
        output_dir: 输出目录路径，默认为None时会自动生成
        task_id: 任务ID，用于生成唯一的输出目录，默认为None时会自动生成

    Returns:
        Dict[str, str]: 时间戳到裁剪后视频路径的映射
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_origin_path):
        raise FileNotFoundError(f"视频文件不存在: {video_origin_path}")

    # 如果未提供task_id，则根据输入生成一个唯一ID
    # if task_id is None:
    #     content_for_hash = f"{video_origin_path}_{json.dumps(tts_path)}"
    #     task_id = hashlib.md5(content_for_hash.encode()).hexdigest()

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "storage", "temp", "clip_video"
        )

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取硬件加速支持
    hwaccel = check_hardware_acceleration()
    hwaccel_args = []
    if hwaccel:
        hwaccel_args = ffmpeg_utils.get_ffmpeg_hwaccel_args()

    video_duration = float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', video_origin_path]))
    audio_duration = float(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', tts_path]))
    speed = video_duration / audio_duration  # 计算速度比例
    # 存储裁剪结果

    output_filename = f"vid_{task_id}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # 构建FFmpeg命令
    ffmpeg_cmd = [
        'ffmpeg',
        '-y', *hwaccel_args,
        '-i', video_origin_path,
        '-filter:v', f'setpts={1 / speed}*PTS',  # 调整视频速度
        '-an',  # 不包含音频
        '-c:v', "h264_videotoolbox" if hwaccel == "videotoolbox" else "libx264", # 重新编码视频（因为修改了速度）
        output_path
    ]

    # 执行FFmpeg命令
    try:

        # logger.debug(f"执行命令: {' '.join(ffmpeg_cmd)}")

        # 在Windows系统上使用UTF-8编码处理输出，避免GBK编码错误
        is_windows = os.name == 'nt'
        if is_windows:
            process = subprocess.run(
                ffmpeg_cmd,
                encoding='utf-8',  # 明确指定编码为UTF-8
                text=True,
                check=True
            )
        else:
            process = subprocess.run(
                ffmpeg_cmd,
                text=True,
                check=True
            )

        result = output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"错误信息: {e.stderr}")
        raise RuntimeError(f"视频裁剪失败: {e.stderr}")

    return result
