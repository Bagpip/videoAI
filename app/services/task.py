import math
import json
import os.path
import re
import traceback
from os import path
from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams, VideoClipParams
from app.services import (llm, material, subtitle, video, voice, audio_merger,
                          subtitle_merger, clip_video, merger_video, update_script, generate_video)
from app.services import state as sm
from app.utils import utils
import streamlit as st
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_dir)  # 直接添加当前目录

import generate_merger
import merger_videov2
from subtitle_merger import parse_time,format_time


def start_subclip(task_id: str, params: VideoClipParams, subclip_path_videos: dict):
    """
    后台任务（自动剪辑视频进行剪辑）
    Args:
        task_id: 任务ID
        params: 视频参数
        subclip_path_videos: 视频片段路径
    """
    global merged_audio_path, merged_subtitle_path

    logger.info(f"\n\n## 开始任务: {task_id}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=0)

    # # 初始化 ImageMagick
    # if not utils.init_imagemagick():
    #     logger.warning("ImageMagick 初始化失败，字幕可能无法正常显示")

    # # tts 角色名称
    # voice_name = voice.parse_voice_name(params.voice_name)
    if st.session_state.get('generate_video_setting'):
        logger.info("\n\n## 1. 加载视频脚本")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        json_path = os.path.join(parent_parent_dir, 'resource', 'scripts', 'video_story.json')
        # 加载原始 JSON 文件

        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 转换为 voice.py 所需的格式
        list_script = []
        for item in data["plot_titles"]:
            list_script.append({
                "_id": item["title"],  # 使用标题作为唯一标识符
                "timestamp": item["timestamp"].replace(",", ":"),  # 替换逗号为冒号（00:00:00,000 → 00:00:00:000）
                "narration": item["content"],  # 使用 content 作为文本
                "OST": 0  # 标记为非背景音乐
            })

        # 保存为新的 JSON 文件（可选）
        voice_script_path = os.path.join(parent_parent_dir, 'resource', 'scripts', 'voice_script.json')
        with open(voice_script_path, "w", encoding="utf-8") as file:
            json.dump(list_script, file, ensure_ascii=False, indent=4)
        with open(voice_script_path, "r", encoding="utf-8") as file:
            list_script = json.load(file)
        task_id = "test_video"  # 任务ID（用于生成输出目录）
        # voice_name = "zh-CN-XiaoxiaoNeural-Female"  # 中文女声语音
        # voice_rate = 1.0  # 语音速率（1.0为正常）
        # voice_pitch = 1.0  # 语音音高（1.0为正常）
        logger.info("\n\n## 2. 生成音频和字幕")
        # 生成音频和字幕
        tts_results = voice.tts_multiple(
            task_id=task_id,
            list_script=list_script,
            voice_name=params.voice_name,
            voice_rate=params.voice_rate,
            voice_pitch=params.voice_pitch,
        )
        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)


    else:
        """
        1. 加载剪辑脚本
        """
        logger.info("\n\n## 1. 加载视频脚本")
        video_script_path = path.join(params.video_clip_json_path)

        if path.exists(video_script_path):
            try:
                with open(video_script_path, "r", encoding="utf-8") as f:
                    list_script = json.load(f)
                    video_list = [i['narration'] for i in list_script]
                    video_ost = [i['OST'] for i in list_script]
                    time_list = [i['timestamp'] for i in list_script]

                    video_script = " ".join(video_list)
                    logger.debug(f"解说完整脚本: \n{video_script}")
                    logger.debug(f"解说 OST 列表: \n{video_ost}")
                    logger.debug(f"解说时间戳列表: \n{time_list}")
            except Exception as e:
                logger.error(f"无法读取视频json脚本，请检查脚本格式是否正确")
                raise ValueError("无法读取视频json脚本，请检查脚本格式是否正确")
        else:
            logger.error(f"video_script_path: {video_script_path} \n\n", traceback.format_exc())
            raise ValueError("解说脚本不存在！请检查配置是否正确。")

        """
        2. 使用 TTS 生成音频素材
        """
        logger.info("\n\n## 2. 根据OST设置生成音频列表")
        # 只为OST=0 or 2的判断生成音频， OST=0 仅保留解说 OST=2 保留解说和原声
        tts_segments = [
            segment for segment in list_script
            if segment['OST'] in [0, 2]
        ]
        logger.debug(f"需要生成TTS的片段数: {len(tts_segments)}")

        tts_results = voice.tts_multiple(
            task_id=task_id,
            list_script=tts_segments,  # 只传入需要TTS的片段
            voice_name=params.voice_name,
            voice_rate=params.voice_rate,
            voice_pitch=params.voice_pitch,
        )

        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # """
    # 3. (可选) 使用 whisper 生成字幕
    # """
    # if merged_subtitle_path is None:
    #     if audio_files:
    #         merged_subtitle_path = path.join(utils.task_dir(task_id), f"subtitle.srt")
    #         subtitle_provider = config.app.get("subtitle_provider", "").strip().lower()
    #         logger.info(f"\n\n使用 {subtitle_provider} 生成字幕")
    #
    #         subtitle.create(
    #             audio_file=merged_audio_path,
    #             subtitle_file=merged_subtitle_path,
    #         )
    #         subtitle_lines = subtitle.file_to_subtitles(merged_subtitle_path)
    #         if not subtitle_lines:
    #             logger.warning(f"字幕文件无效: {merged_subtitle_path}")
    #
    # sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)
    if st.session_state.get('generate_video_setting'):
        logger.info("\n\n## 3. 匹配视频音频长度")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        videos_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'output_videos')
        # 获取视频目录列表
        video_list = [os.path.join(videos_path, f) for f in os.listdir(videos_path)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
        tts_path = os.path.join(parent_parent_dir, 'storage', 'tasks', task_id)
        # 获取音频和字幕列表
        tts_list = [os.path.join(tts_path, f) for f in os.listdir(tts_path)
                       if f.lower().endswith(('.mp3', '.wav'))]
        srt_list = [os.path.join(tts_path, f) for f in os.listdir(tts_path)
                       if f.lower().endswith('.srt')]

        if len(video_list) != len(tts_list):
            raise ValueError(f"视频和音频数量不匹配！视频: {len(video_list)}, 音频: {len(tts_list)}")

        chip_video_list = []
        for index in range(len(video_list)):
            chip_result = generate_merger.generate_video_tts_srt(video_origin_path = video_list[index],
                                   tts_path = tts_list[index],
                                   task_id = index)
            chip_video_list.append(chip_result)


        logger.info("\n\n## 4. 合并音频和字幕")
        from pydub import AudioSegment

        def simple_merge_mp3(files: list, output_path: str):
            """简单合并多个MP3文件"""
            combined = AudioSegment.empty()
            for file in files:
                sound = AudioSegment.from_mp3(file)
                combined += sound
            combined.export(output_path, format="mp3")
            return output_path

        # 使用示例
        merged_audio_path = simple_merge_mp3(tts_list, os.path.join(parent_parent_dir, 'storage', 'temp', 'merge', "merged_audio.mp3"))
        print(f"合并完成，输出文件: {merged_audio_path}")

        import re
        from datetime import timedelta
        def merge_srt_files(srt_files, output_file="merged.srt"):
            """合并多个 SRT 文件"""
            merged_blocks = []
            current_time = timedelta()
            subtitle_index = 1

            for file in srt_files:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # 分割字幕块（以空行分隔）
                blocks = re.split(r'\n\s*\n', content)

                for block in blocks:
                    lines = block.strip().split('\n')
                    if len(lines) < 3:  # 确保是有效的字幕块
                        continue

                    # 解析时间轴
                    time_line = lines[1]
                    start_str, end_str = time_line.split(' --> ')
                    start_time = parse_time(start_str)
                    end_time = parse_time(end_str)

                    # 调整时间（累加偏移量）
                    new_start = current_time + start_time
                    new_end = current_time + end_time

                    # 重新编号并存储
                    merged_blocks.append(
                        f"{subtitle_index}\n"
                        f"{format_time(new_start)} --> {format_time(new_end)}\n"
                        + "\n".join(lines[2:])
                    )
                    subtitle_index += 1

                # 更新当前时间（下一个文件的起始时间 = 当前文件的结束时间）
                last_block = blocks[-1]
                last_end_str = last_block.split('\n')[1].split(' --> ')[1]
                current_time += parse_time(last_end_str)

            # 写入合并后的文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(merged_blocks))
            return output_file

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
        srt_path = os.path.join(parent_parent_dir, 'resource', 'srt', 'merged.srt')
        merged_subtitle_path = merge_srt_files(srt_list, output_file=srt_path)

        chip_video_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'clip_video')
        chip_video_path = [os.path.join(chip_video_path, f) for f in os.listdir(chip_video_path)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
        final_video_paths = []
        combined_video_paths = []
        combined_video_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'merge', "merged_video_01.mp4")
        combined_video_path = merger_videov2.merge_videos_without_audio(
            output_video_path=combined_video_path,
            video_paths=chip_video_path,
            video_aspect=params.video_aspect
        )
        output_video_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'merge', "merged_video_02.mp4")





    else:
        """
        3. 裁剪视频 - 将超出音频长度的视频进行裁剪
        """
        logger.info("\n\n## 3. 裁剪视频")
        video_clip_result = clip_video.clip_video(params.video_origin_path, tts_results)
        # 更新 list_script 中的时间戳
        tts_clip_result = {tts_result['_id']: tts_result['audio_file'] for tts_result in tts_results}
        subclip_clip_result = {
            tts_result['_id']: tts_result['subtitle_file'] for tts_result in tts_results
        }
        new_script_list = update_script.update_script_timestamps(list_script, video_clip_result, tts_clip_result, subclip_clip_result)

        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=60)

        """
        4. 合并音频和字幕
        """
        logger.info("\n\n## 4. 合并音频和字幕")
        total_duration = sum([script["duration"] for script in new_script_list])
        if tts_segments:
            try:
                # 合并音频文件
                merged_audio_path = audio_merger.merge_audio_files(
                    task_id=task_id,
                    total_duration=total_duration,
                    list_script=new_script_list
                )
                logger.info(f"音频文件合并成功->{merged_audio_path}")
                # 合并字幕文件
                merged_subtitle_path = subtitle_merger.merge_subtitle_files(new_script_list)
                logger.info(f"字幕文件合并成功->{merged_subtitle_path}")
            except Exception as e:
                logger.error(f"合并音频文件失败: {str(e)}")
        else:
            logger.warning("没有需要合并的音频/字幕")
            merged_audio_path = ""
            merged_subtitle_path = ""

        """
        5. 合并视频
        """
        final_video_paths = []
        combined_video_paths = []

        combined_video_path = path.join(utils.task_dir(task_id), f"merger.mp4")
        logger.info(f"\n\n## 5. 合并视频: => {combined_video_path}")
        # 如果 new_script_list 中没有 video，则使用 subclip_path_videos 中的视频
        video_clips = [new_script['video'] if new_script.get('video') else subclip_path_videos.get(new_script.get('_id', '')) for new_script in new_script_list]

        merger_video.combine_clip_videos(
            output_video_path=combined_video_path,
            video_paths=video_clips,
            video_ost_list=video_ost,
            video_aspect=params.video_aspect,
            threads=params.n_threads
        )
        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=80)

        """
        6. 合并字幕/BGM/配音/视频
        """
        output_video_path = path.join(utils.task_dir(task_id), f"combined.mp4")
    logger.info(f"\n\n## 6. 最后一步: 合并字幕/BGM/配音/视频 -> {output_video_path}")

    # bgm_path = '/Users/apple/Desktop/home/NarratoAI/resource/songs/bgm.mp3'
    bgm_path = utils.get_bgm_file()

    # 调用示例
    options = {
        'voice_volume': params.tts_volume,  # 配音音量
        'bgm_volume': params.bgm_volume,  # 背景音乐音量
        'original_audio_volume': params.original_volume,  # 视频原声音量，0表示不保留
        'keep_original_audio': True,  # 是否保留原声
        'subtitle_font': params.font_name,  # 这里使用相对字体路径，会自动在 font_dir() 目录下查找
        'subtitle_font_size': params.font_size,
        'subtitle_color': params.text_fore_color,
        'subtitle_bg_color': None,  # 直接使用None表示透明背景
        'subtitle_position': params.subtitle_position,
        'custom_position': params.custom_position,
        'threads': params.n_threads
    }
    generate_video.merge_materials(
        video_path=combined_video_path,
        audio_path=merged_audio_path,
        subtitle_path=merged_subtitle_path,
        bgm_path=bgm_path,
        output_path=output_video_path,
        options=options
    )

    final_video_paths.append(output_video_path)
    combined_video_paths.append(combined_video_path)

    logger.success(f"任务 {task_id} 已完成, 生成 {len(final_video_paths)} 个视频.")

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths
    }
    sm.state.update_task(task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs)
    return kwargs


def validate_params(video_path, audio_path, output_file, params):
    """
    验证输入参数
    Args:
        video_path: 视频文件路径
        audio_path: 音频文件路径（可以为空字符串）
        output_file: 输出文件路径
        params: 视频参数

    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 参数无效时抛出
    """
    if not video_path:
        raise ValueError("视频路径不能为空")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    # 如果提供了音频路径，则验证文件是否存在
    if audio_path and not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        
    if not output_file:
        raise ValueError("输出文件路径不能为空")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not params:
        raise ValueError("视频参数不能为空")


if __name__ == "__main__":
    task_id = "demo"
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # json_path = os.path.join(parent_parent_dir, 'resource', 'scripts', 'voice_script.json')
    # 提前裁剪是为了方便检查视频
    subclip_path_videos = {
        1: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-00-05-390@00-00-57-980.mp4',
        2: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-00-28-900@00-00-43-700.mp4',
        3: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-01-17-840@00-01-27-600.mp4',
        4: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-02-35-460@00-02-52-380.mp4',
        5: '/Users/apple/Desktop/home/NarratoAI/storage/temp/clip_video/113343d127b5a09d0bf84b68bd1b3b97/vid_00-06-59-520@00-07-29-500.mp4',
    }

    params = VideoClipParams(
        video_clip_json_path="/Users/apple/Desktop/home/NarratoAI/resource/scripts/2025-0507-223311.json",
        video_origin_path="/Users/apple/Desktop/home/NarratoAI/resource/videos/merged_video_4938.mp4",
    )
    start_subclip(task_id, params, subclip_path_videos)
