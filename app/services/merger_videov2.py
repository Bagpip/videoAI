#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import shutil
import subprocess
from enum import Enum
from typing import List, Tuple
from loguru import logger


class VideoAspect(Enum):
    """视频宽高比枚举"""
    landscape = "16:9"  # 横屏 16:9
    portrait = "9:16"  # 竖屏 9:16
    square = "1:1"  # 方形 1:1

    def to_resolution(self) -> Tuple[int, int]:
        """根据宽高比返回标准分辨率"""
        if self == VideoAspect.portrait:
            return 1080, 1920  # 竖屏 9:16
        elif self == VideoAspect.landscape:
            return 1920, 1080  # 横屏 16:9
        elif self == VideoAspect.square:
            return 1080, 1080  # 方形 1:1
        else:
            return 1080, 1920  # 默认竖屏


def check_ffmpeg_installation() -> bool:
    """检查ffmpeg是否已安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("ffmpeg未安装或不在系统PATH中，请安装ffmpeg")
        return False


def create_ffmpeg_concat_file(video_paths: List[str], concat_file_path: str) -> str:
    """创建ffmpeg合并所需的concat文件"""
    with open(concat_file_path, 'w', encoding='utf-8') as f:
        for video_path in video_paths:
            abs_path = os.path.abspath(video_path)
            if os.name == 'nt':  # Windows系统
                abs_path = abs_path.replace('\\', '/')
            else:  # Unix/Mac系统
                abs_path = abs_path.replace('\\', '\\\\').replace(':', '\\:')
            abs_path = abs_path.replace("'", "\\'")
            f.write(f"file '{abs_path}'\n")
    return concat_file_path


def merge_videos_without_audio(
        output_video_path: str,
        video_paths: List[str],
        video_aspect: VideoAspect = VideoAspect.portrait,
        threads: int = 4
) -> str:
    """
    合并无音频视频

    Args:
        output_video_path: 合并后的视频路径
        video_paths: 要合并的视频路径列表
        video_aspect: 视频宽高比
        threads: 使用的线程数
    """
    # 检查ffmpeg是否安装
    if not check_ffmpeg_installation():
        raise RuntimeError("未找到ffmpeg，请先安装")

    # 准备输出目录
    output_dir = os.path.dirname(output_video_path)
    os.makedirs(output_dir, exist_ok=True)

    # 创建临时目录
    temp_dir = os.path.join(output_dir, "temp_videos")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 创建concat文件
        concat_file = os.path.join(temp_dir, "concat_list.txt")
        create_ffmpeg_concat_file(video_paths, concat_file)

        # 获取目标分辨率
        video_width, video_height = video_aspect.to_resolution()

        # 构建合并命令
        merge_cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_file,
            '-vf', f'scale={video_width}:{video_height}:force_original_aspect_ratio=decrease,'
                   f'pad={video_width}:{video_height}:(ow-iw)/2:(oh-ih)/2',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-profile:v', 'high',
            '-r', '30',  # 设置帧率为30fps
            '-an',  # 确保不包含音频
            '-threads', str(threads),
            output_video_path
        ]

        # 执行合并命令
        subprocess.run(merge_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"视频合并成功: {output_video_path}")

        return output_video_path

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"合并视频失败: {error_msg}")
        raise RuntimeError(f"无法合并视频: {error_msg}")
    finally:
        # 清理临时文件
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("已清理临时文件")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {str(e)}")


if __name__ == '__main__':
    # 示例用法
    video_paths = [
        '/path/to/video1.mp4',
        '/path/to/video2.mp4',
        '/path/to/video3.mp4'
    ]

    merge_videos_without_audio(
        output_video_path="/path/to/output/merged_video.mp4",
        video_paths=video_paths,
        video_aspect=VideoAspect.portrait
    )