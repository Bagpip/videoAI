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



import os
import cv2
import numpy as np
from moviepy import AudioFileClip, concatenate_audioclips
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import tempfile
import math
from typing import List, Tuple, Optional


class AdvancedSlideshowGenerator:
    def __init__(self, output_quality='high'):
        self.quality_profiles = {
            'low': {'codec': 'libx264', 'crf': 28, 'preset': 'fast'},
            'medium': {'codec': 'libx264', 'crf': 23, 'preset': 'medium'},
            'high': {'codec': 'libx264', 'crf': 18, 'preset': 'slow'}
        }
        self.output_quality = output_quality

    def _apply_ken_burns_effect(self, img: np.ndarray, duration: float,
                                zoom_rate=0.1, pan_range=0.2) -> List[np.ndarray]:
        """应用Ken Burns效果"""
        height, width = img.shape[:2]
        frames = []

        # 随机选择起始和结束位置
        start_scale = 1.0 + zoom_rate * np.random.random()
        end_scale = 1.0 + zoom_rate * np.random.random()

        # 随机选择平移方向
        start_x = pan_range * width * (np.random.random() - 0.5)
        start_y = pan_range * height * (np.random.random() - 0.5)
        end_x = pan_range * width * (np.random.random() - 0.5)
        end_y = pan_range * height * (np.random.random() - 0.5)

        # 生成每一帧
        for t in np.linspace(0, 1, int(duration * 24)):  # 24fps
            ease_t = 0.5 - 0.5 * math.cos(t * math.pi)
            current_scale = start_scale + (end_scale - start_scale) * ease_t
            current_x = start_x + (end_x - start_x) * ease_t
            current_y = start_y + (end_y - start_y) * ease_t

            M = np.float32([
                [current_scale, 0, (1 - current_scale) * width / 2 + current_x],
                [0, current_scale, (1 - current_scale) * height / 2 + current_y]
            ])

            transformed = cv2.warpAffine(img, M, (width, height),
                                         borderMode=cv2.BORDER_REFLECT)
            frames.append(transformed)

        return frames

    def _process_image_audio_pair(self, img_path: str, audio_path: str,
                                  output_dir: str, idx: int) -> Optional[Tuple[str, str, int]]:
        """
        处理单个图片-音频对
        返回: (临时视频路径, 音频路径, 原始索引) 或 None
        """
        try:
            # 读取音频
            audio_clip = AudioFileClip(audio_path)
            duration = audio_clip.duration

            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图片: {img_path}")

            # 应用效果
            frames = self._apply_ken_burns_effect(img, duration)

            # 写入临时视频
            temp_video = os.path.join(output_dir, f"temp_{idx}.mp4")
            height, width = img.shape[:2]

            # 使用FFmpeg写入视频
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'image2pipe',
                '-vcodec', 'png',
                '-r', '24',
                '-s', f'{width}x{height}',
                '-i', '-',
                '-c:v', self.quality_profiles[self.output_quality]['codec'],
                '-crf', str(self.quality_profiles[self.output_quality]['crf']),
                '-preset', self.quality_profiles[self.output_quality]['preset'],
                '-pix_fmt', 'yuv420p',
                temp_video
            ]

            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

            for frame in frames:
                _, buffer = cv2.imencode('.png', frame)
                process.stdin.write(buffer.tobytes())

            process.stdin.close()
            process.wait()

            return temp_video, audio_path, idx

        except Exception as e:
            print(f"处理失败 {img_path}: {str(e)}")
            return None

    def generate(self, image_folder: str, audio_folder: str, output_video: str,
                 max_workers: int = 4):
        """生成高质量幻灯片视频"""
        # 1. 匹配文件并保持顺序
        image_files = sorted([f for f in os.listdir(image_folder)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                             key=lambda x: int(''.join(filter(str.isdigit, x))))

        file_pairs = []
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            audio_file = f"{base_name}.mp3"
            audio_path = os.path.join(audio_folder, audio_file)
            if os.path.exists(audio_path):
                file_pairs.append((
                    os.path.join(image_folder, img_file),
                    audio_path,
                    len(file_pairs)  # 原始索引
                ))

        if not file_pairs:
            raise ValueError("没有找到匹配的图片和音频文件对")

        # 2. 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 3. 并行处理
            temp_videos = [None] * len(file_pairs)
            audio_paths = [None] * len(file_pairs)
            success_count = 0

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for img_path, audio_path, idx in file_pairs:
                    futures.append(
                        executor.submit(
                            self._process_image_audio_pair,
                            img_path, audio_path, temp_dir, idx
                        )
                    )

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="处理图片-音频对"):
                    result = future.result()
                    if result is not None:
                        video, audio, idx = result
                        temp_videos[idx] = video
                        audio_paths[idx] = audio
                        success_count += 1

            if success_count == 0:
                raise ValueError("所有处理都失败了，请检查输入文件和错误日志")

            # 4. 过滤有效结果
            valid_videos = [v for v in temp_videos if v is not None]
            valid_audios = [a for a in audio_paths if a is not None]

            # 5. 合并音频
            try:
                audio_clips = [AudioFileClip(a) for a in valid_audios]
                final_audio = concatenate_audioclips(audio_clips)
                temp_audio = os.path.join(temp_dir, "final_audio.mp3")
                final_audio.write_audiofile(temp_audio)

                # 6. 合并视频
                with open(os.path.join(temp_dir, "filelist.txt"), 'w') as f:
                    for video in valid_videos:
                        f.write(f"file '{video}'\n")

                merge_cmd = [
                    'ffmpeg', '-y',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', os.path.join(temp_dir, "filelist.txt"),
                    '-i', temp_audio,
                    '-c:v', self.quality_profiles[self.output_quality]['codec'],
                    '-crf', str(self.quality_profiles[self.output_quality]['crf']),
                    '-preset', self.quality_profiles[self.output_quality]['preset'],
                    '-c:a', 'aac',
                    '-movflags', '+faststart',
                    '-shortest',
                    output_video
                ]

                subprocess.run(merge_cmd, check=True)

                print(f"成功生成视频: {output_video}")
                print(f"成功处理 {success_count}/{len(file_pairs)} 个文件")

            except Exception as e:
                raise RuntimeError(f"最终合并失败: {str(e)}")


# 使用示例
if __name__ == "__main__":
    generator = AdvancedSlideshowGenerator(output_quality='high')
    generator.generate(
        image_folder=r"D:\newbegin\NarratoAI\test_resource\img",
        audio_folder=r"D:\newbegin\NarratoAI\test_resource\audio",
        output_video=r"D:\newbegin\NarratoAI\output_video001.mp4",
        max_workers=4
    )
# 使用示例
# if __name__ == "__main__":
#     generator = AdvancedSlideshowGenerator(output_quality='high')
#     generator.generate(
#         image_folder=r"D:\newbegin\NarratoAI\test_resource\img",
#         audio_folder=r"D:\newbegin\NarratoAI\test_resource\audio",
#         output_video=r"D:\newbegin\NarratoAI\output_video001.mp4",
#         max_workers=4
#     )
