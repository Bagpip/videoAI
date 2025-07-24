import os
import cv2
import numpy as np
from moviepy import AudioFileClip, concatenate_audioclips
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import tempfile
from loguru import logger
import math
from typing import List, Tuple, Optional, Union
import contextlib  # 添加这行导入
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import sys
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_dir)  # 直接添加当前目录


class AdvancedSlideshowGenerator:
    def __init__(self, output_quality='high', target_aspect=None, outro_video_path=None,
                 text_overlay=None, sticker_overlay=None, subtitle_settings=None):
        self.quality_profiles = {
            'low': {'codec': 'libx264', 'crf': 28, 'preset': 'fast'},
            'medium': {'codec': 'libx264', 'crf': 23, 'preset': 'medium'},
            'high': {'codec': 'libx264', 'crf': 18, 'preset': 'slow'}
        }
        self.output_quality = output_quality
        self.target_aspect = target_aspect
        self.outro_video_path = outro_video_path
        self.text_overlay = text_overlay or {}
        self.sticker_overlay = sticker_overlay or {}
        self.subtitle_settings = subtitle_settings or {
            'font_size': 24,
            'color': (255, 255, 255),
            'outline_color': (0, 0, 0),
            'outline_width': 2,
            'pos_x_ratio': 0.5,
            'pos_y_ratio': 0.85,
            'alignment': 'center'
        }

    def _parse_srt(self, srt_path: str) -> List[Tuple[float, float, str]]:
        """解析SRT字幕文件"""
        subtitles = []
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_sub = None
            for line in lines:
                line = line.strip()
                if not line:
                    if current_sub:
                        subtitles.append(current_sub)
                    current_sub = None
                elif '-->' in line:
                    start, end = line.split('-->')
                    start = start.strip().replace(',', '.')
                    end = end.strip().replace(',', '.')
                    start_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(start.split(':'))))
                    end_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(end.split(':'))))
                    current_sub = (start_time, end_time, '')
                elif current_sub and current_sub[2] == '':
                    current_sub = (current_sub[0], current_sub[1], line)
                elif current_sub:
                    current_sub = (current_sub[0], current_sub[1], current_sub[2] + '\n' + line)

            if current_sub:
                subtitles.append(current_sub)
        except Exception as e:
            logger.warning(f"解析字幕文件失败 {srt_path}: {str(e)}")
        return subtitles

    def _draw_text_with_outline(self, draw, text, position, font, text_color, outline_color, outline_width):
        """绘制带描边的文本"""
        x, y = position
        # 绘制描边
        for dx in [-outline_width, 0, outline_width]:
            for dy in [-outline_width, 0, outline_width]:
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # 绘制文本
        draw.text(position, text, font=font, fill=text_color)

    def _add_subtitles(self, frame: np.ndarray, current_time: float,
                       subtitles: List[Tuple[float, float, str]]) -> np.ndarray:
        """在帧上添加字幕"""
        if not subtitles:
            return frame

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 查找当前时间应该显示的字幕
        active_subtitles = [sub for sub in subtitles if sub[0] <= current_time <= sub[1]]
        if not active_subtitles:
            return frame

        # 获取字幕设置
        font_size = self.subtitle_settings['font_size']
        color = self.subtitle_settings['color']
        outline_color = self.subtitle_settings['outline_color']
        outline_width = self.subtitle_settings['outline_width']
        pos_x_ratio = self.subtitle_settings['pos_x_ratio']
        pos_y_ratio = self.subtitle_settings['pos_y_ratio']
        alignment = self.subtitle_settings['alignment']

        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # 计算文本位置
        img_width, img_height = img_pil.size
        x = int(img_width * pos_x_ratio)
        y = int(img_height * pos_y_ratio)

        # 绘制所有活动字幕
        for sub in active_subtitles:
            text = sub[2]

            # 计算文本大小
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # 根据对齐方式调整位置
            if alignment == 'center':
                x_pos = x - text_width // 2
            elif alignment == 'left':
                x_pos = x
            else:  # right
                x_pos = x - text_width

            # 绘制带描边的文本
            self._draw_text_with_outline(
                draw, text, (x_pos, y),
                font, color, outline_color, outline_width
            )

            # 多行文本需要调整y位置
            y += text_height + 5

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _apply_ken_burns_effect_single_frame(self, img: np.ndarray, t: float, duration: float,
                                             zoom_rate=0.1, pan_range=0.2) -> np.ndarray:
        """应用Ken Burns效果到单帧"""
        height, width = img.shape[:2]

        # 计算当前时间比例
        ease_t = 0.5 - 0.5 * math.cos(t / duration * math.pi)

        # 随机选择起始和结束位置
        start_scale = 1.0 + zoom_rate * np.random.random()
        end_scale = 1.0 + zoom_rate * np.random.random()
        current_scale = start_scale + (end_scale - start_scale) * ease_t

        # 随机选择平移方向
        start_x = pan_range * width * (np.random.random() - 0.5)
        start_y = pan_range * height * (np.random.random() - 0.5)
        end_x = pan_range * width * (np.random.random() - 0.5)
        end_y = pan_range * height * (np.random.random() - 0.5)
        current_x = start_x + (end_x - start_x) * ease_t
        current_y = start_y + (end_y - start_y) * ease_t

        # 应用变换
        M = np.float32([
            [current_scale, 0, (1 - current_scale) * width / 2 + current_x],
            [0, current_scale, (1 - current_scale) * height / 2 + current_y]
        ])

        return cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REFLECT)

    def _create_trapezoid_sticker(self, width=240, height=100):
        """创建等腰梯形贴纸，两腰延长线相交成90度"""
        # 计算梯形参数：下底(width)，高度(height)，上底 = width - 2*height
        # 确保上底不小于最小宽度(20px)
        top_width = max(20, width - 2 * height)

        # 创建紧凑的画布，只比梯形稍大一点
        canvas_width = width
        canvas_height = height
        sticker = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sticker)

        # 定义梯形四点坐标（从左上角开始顺时针）
        points = [
            ((width - top_width) // 2, 0),  # 左上
            ((width + top_width) // 2, 0),  # 右上
            (width, height),  # 右下
            (0, height)  # 左下
        ]

        # 绘制梯形
        draw.polygon(points, fill=(255, 235, 59, 255))  # 黄色填充

        # 添加阴影效果（在梯形下方）
        shadow = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 30))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.polygon(points, fill=(0, 0, 0, 30))
        sticker = Image.alpha_composite(sticker, shadow)

        return sticker, points

    def _add_sticker_with_text(self, frame: np.ndarray) -> np.ndarray:
        """在帧上添加梯形贴纸和文本，旋转后裁剪多余区域并紧贴右上角"""
        if not self.sticker_overlay:
            return frame

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_width, img_height = img_pil.size

        # 获取配置参数
        sticker_width = self.sticker_overlay.get('width', 240)
        sticker_height = self.sticker_overlay.get('height', 100)
        sticker_text = self.sticker_overlay.get('text', '')
        text_color = self.sticker_overlay.get('text_color', (0, 0, 0))
        text_size = self.sticker_overlay.get('text_size', 24)

        # 创建梯形贴纸
        sticker, trapezoid_points = self._create_trapezoid_sticker(sticker_width, sticker_height)

        # 创建文本贴纸
        text_sticker = Image.new('RGBA', sticker.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_sticker)

        try:
            font = ImageFont.truetype("simhei.ttf", text_size)
        except:
            font = ImageFont.load_default()

        # 计算文本位置（梯形中心）
        text_bbox = font.getbbox(sticker_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        center_x = (trapezoid_points[0][0] + trapezoid_points[1][0]) / 1.7
        center_y = sticker_height / 2

        text_position = (
            center_x - text_width / 2,
            center_y - text_height / 2
        )

        draw.text(text_position, sticker_text, font=font, fill=text_color)

        # 旋转角度45度，围绕梯形右下角旋转
        angle = 45
        rotation_center = (sticker_width, sticker_height)  # 右下角作为旋转中心

        # 旋转贴纸和文本贴纸
        rotated_sticker = sticker.rotate(
            -angle,
            expand=True,
            center=rotation_center,
            resample=Image.BICUBIC
        )

        rotated_text = text_sticker.rotate(
            -angle,
            expand=True,
            center=rotation_center,
            resample=Image.BICUBIC
        )

        # 裁剪旋转后的图像，只保留右下部分（确保贴纸完整）
        # 获取旋转后图像的边界框
        bbox = rotated_sticker.getbbox()
        if bbox:
            # 只保留右半部分和下半部分
            crop_box = (
                bbox[0],  # 从中间开始裁剪x
                bbox[1],  # 保持完整的y
                bbox[2],  # 保留右边界
                bbox[3]  # 保留下边界
            )
            rotated_sticker = rotated_sticker.crop(crop_box)
            rotated_text = rotated_text.crop(crop_box)

        # 计算贴纸在右上角的位置（紧贴边缘）
        pos_x = img_width - rotated_sticker.width  # 紧贴右边界
        pos_y = 0  # 紧贴上边界

        # 合成贴纸和文本
        img_pil.paste(rotated_sticker, (pos_x, pos_y), rotated_sticker)
        img_pil.paste(rotated_text, (pos_x, pos_y), rotated_text)

        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def _get_char_size(self, font, char):
        """获取单个字符的尺寸（兼容Pillow新版本）"""
        # 方法1：使用getbbox（推荐）
        bbox = font.getbbox(char)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]  # width, height

        # 或者方法2：使用getlength（仅宽度）
        # width = int(font.getlength(char))
        # height = font.size  # 近似高度
        # return width, height

    def _draw_text_with_spacing(self, draw, text, position, font, color, letter_spacing=0):
        """支持字间距的文本绘制方法（兼容Pillow新版本）"""
        x, y = position
        total_width = 0
        char_sizes = []

        # 先计算所有字符的尺寸和总宽度
        for char in text:
            char_width, char_height = self._get_char_size(font, char)
            char_sizes.append((char_width, char_height))
            total_width += char_width + letter_spacing

        # 调整起始x位置以实现居中
        if self.text_overlay.get('pos_x_ratio', 0.5) == 0.5:
            x -= total_width // 2

        # 逐个字符绘制
        for i, char in enumerate(text):
            char_width, char_height = char_sizes[i]
            draw.text((x, y), char, font=font, fill=color)
            x += char_width + letter_spacing

    def _add_text_overlay(self, frame: np.ndarray) -> np.ndarray:
        """在帧上添加可自定义位置和字间距的文本"""
        frame = self._add_sticker_with_text(frame)  # 先添加贴纸

        if not self.text_overlay or not self.text_overlay.get('text'):
            return frame

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        text = self.text_overlay['text']
        font_size = self.text_overlay.get('font_size', 30)
        alpha = self.text_overlay.get('alpha', 0.75)
        color = self.text_overlay.get('color', (255, 255, 255))
        letter_spacing = self.text_overlay.get('letter_spacing', 1)

        # 位置比例参数
        pos_x_ratio = self.text_overlay.get('pos_x_ratio', 0.5)
        pos_y_ratio = self.text_overlay.get('pos_y_ratio', 0.83)

        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # 计算文本高度（使用第一个字符的高度作为参考）
        _, text_height = self._get_char_size(font, text[0] if text else ' ')
        img_width, img_height = img_pil.size

        # 计算位置
        x = int(img_width * pos_x_ratio)
        y = int(img_height * pos_y_ratio - text_height)

        # 绘制带字间距的文本
        self._draw_text_with_spacing(
            draw, text, (x, y),
            font, color,
            letter_spacing=letter_spacing
        )

        frame_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if alpha < 1.0:
            frame_with_text = cv2.addWeighted(frame_with_text, alpha, frame, 1 - alpha, 0)

        return frame_with_text

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
            # 添加文本
            transformed = self._add_text_overlay(transformed)

            frames.append(transformed)

        return frames

    def _adjust_aspect_ratio(self, video_path: str, output_path: str) -> str:
        """调整视频比例为指定的宽高比(9:16或16:9)，通过添加黑边实现"""
        if not self.target_aspect:
            return video_path

        # 获取视频原始宽高
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.check_output(cmd).decode('utf-8').strip()
        width, height = map(int, result.split(','))

        # 计算目标宽高
        if self.target_aspect == "9:16":
            # 竖屏9:16 - 保持高度不变，调整宽度
            target_height = height
            target_width = int(height * 9 / 16)
            # 确保新宽度不小于原始宽度
            if target_width < width:
                # 如果目标宽度小于原始宽度，改为保持宽度不变，调整高度
                target_width = width
                target_height = int(width * 16 / 9)
        elif self.target_aspect == "16:9":
            # 横屏16:9 - 保持宽度不变，调整高度
            target_width = width
            target_height = int(width * 9 / 16)
            # 确保新高度不小于原始高度
            if target_height < height:
                # 如果目标高度小于原始高度，改为保持高度不变，调整宽度
                target_height = height
                target_width = int(height * 16 / 9)
        else:
            raise ValueError("不支持的宽高比，只能是'9:16'或'16:9'")

        # 构建FFmpeg命令
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,'
                   f'pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:black',
            '-c:a', 'copy',
            output_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"调整视频比例失败: {e.stderr}")
            raise RuntimeError(f"调整视频比例失败: {e.stderr}")

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

            # 检查是否有对应的字幕文件
            srt_path = os.path.splitext(audio_path)[0] + '.srt'
            subtitles = []
            if os.path.exists(srt_path):
                subtitles = self._parse_srt(srt_path)

            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图片: {img_path}")

            # 应用效果
            frames = self._apply_ken_burns_effect(img, duration)

            frame_times = np.linspace(0, duration, len(frames))
            final_frames = []
            for i, (frame, t) in enumerate(zip(frames, frame_times)):
                # 添加字幕(如果有)
                if subtitles:
                    frame = self._add_subtitles(frame, t, subtitles)
                # 添加固定文本和贴纸
                frame = self._add_text_overlay(frame)
                final_frames.append(frame)

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

            for frame in final_frames:
                _, buffer = cv2.imencode('.png', frame)
                process.stdin.write(buffer.tobytes())

            process.stdin.close()
            process.wait()

            return temp_video, audio_path, idx

        except Exception as e:
            print(f"处理失败 {img_path}: {str(e)}")
            return None

    def _add_outro_video(self, main_video_path: str, output_path: str) -> str:
        """在结尾添加 outro 视频，确保视频和音频都正确合并"""
        if not self.outro_video_path:
            return main_video_path

        # 确保outro视频存在
        if not os.path.exists(self.outro_video_path):
            logger.warning(f"结尾视频不存在: {self.outro_video_path}")
            return main_video_path

        try:
            # 获取主视频的分辨率
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
                main_video_path
            ]
            result = subprocess.check_output(cmd).decode('utf-8').strip()
            main_width, main_height = map(int, result.split(','))

            # 调整outro视频的分辨率与主视频一致
            temp_outro = os.path.join(os.path.dirname(output_path), "temp_outro.mp4")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', self.outro_video_path,
                '-vf', f'scale={main_width}:{main_height}:force_original_aspect_ratio=decrease,'
                       f'pad={main_width}:{main_height}:(ow-iw)/2:(oh-ih)/2:black',
                '-c:a', 'copy',
                temp_outro
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # 使用复杂滤镜合并视频和音频
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', main_video_path,
                '-i', temp_outro,
                '-filter_complex',
                '[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]',
                '-map', '[v]',
                '-map', '[a]',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                output_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            # 验证输出文件
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("合并后的视频文件为空或不存在")

            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"添加结尾视频失败: {e.stderr}")
            raise RuntimeError(f"添加结尾视频失败: {e.stderr}")
        finally:
            if 'temp_outro' in locals() and os.path.exists(temp_outro):
                os.unlink(temp_outro)

    def _extract_audio_and_silent_video(self, video_path: str, audio_output_path: str = None,
                                        silent_video_output_path: str = None) -> Tuple[str, str]:
        """
        提取视频的音频和无声视频
        返回: (音频路径, 无声视频路径)
        """
        try:
            # 确保输出目录存在
            if audio_output_path:
                os.makedirs(os.path.dirname(audio_output_path) or os.path.dirname(audio_output_path), exist_ok=True)
            if silent_video_output_path:
                os.makedirs(os.path.dirname(silent_video_output_path) or os.path.dirname(silent_video_output_path),
                            exist_ok=True)

            # 提取音频
            audio_path = audio_output_path or os.path.splitext(video_path)[0] + '_audio.mp3'
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-q:a', '0',
                '-map', 'a',
                audio_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            # 提取无声视频
            silent_video_path = silent_video_output_path or os.path.splitext(video_path)[0] + '_silent.mp4'
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-c:v', 'copy',
                '-an',  # 禁用音频
                silent_video_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            return audio_path, silent_video_path

        except subprocess.CalledProcessError as e:
            logger.error(f"提取音频或无声视频失败: {e.stderr}")
            raise RuntimeError(f"提取音频或无声视频失败: {e.stderr}")

    def generate(self, image_folder: str, audio_folder: str, output_video: str,
             max_workers: int = 4, extract_audio_path: str = None,
             extract_silent_video_path: str = None):
        """生成高质量幻灯片视频"""
        # 确保输出目录存在
        output_dir = os.path.dirname(output_video)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

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
            try:
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
                temp_main_video = os.path.join(temp_dir, "main_video.mp4")
                temp_audio = os.path.join(temp_dir, "final_audio.mp3")
                temp_aspect_video = os.path.join(temp_dir, "aspect_video.mp4")
                final_output = os.path.join(temp_dir, "final_output.mp4")

                try:
                    audio_clips = [AudioFileClip(a) for a in valid_audios]
                    final_audio = concatenate_audioclips(audio_clips)
                    final_audio.write_audiofile(temp_audio)

                    # 合并视频
                    with open(os.path.join(temp_dir, "filelist.txt"), 'w', encoding='utf-8') as f:
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
                        temp_main_video
                    ]
                    subprocess.run(merge_cmd, check=True)

                    # 验证主视频是否生成成功
                    if not os.path.exists(temp_main_video) or os.path.getsize(temp_main_video) == 0:
                        raise RuntimeError("主视频生成失败，文件为空或不存在")

                    # 调整宽高比
                    video_to_use = temp_main_video
                    if self.target_aspect:
                        self._adjust_aspect_ratio(temp_main_video, temp_aspect_video)
                        if os.path.exists(temp_aspect_video) and os.path.getsize(temp_aspect_video) > 0:
                            video_to_use = temp_aspect_video
                        else:
                            logger.warning("宽高比调整失败，使用原始视频")
                            video_to_use = temp_main_video

                    # 添加结尾视频
                    if self.outro_video_path:
                        self._add_outro_video(video_to_use, final_output)
                        if os.path.exists(final_output) and os.path.getsize(final_output) > 0:
                            final_video = final_output
                        else:
                            logger.warning("添加结尾视频失败，使用主视频")
                            final_video = video_to_use
                    else:
                        final_video = video_to_use

                    # 移动最终文件到目标位置
                    if os.path.exists(final_video) and os.path.getsize(final_video) > 0:
                        shutil.move(final_video, output_video)
                        print(f"成功生成视频: {output_video}")
                        print(f"成功处理 {success_count}/{len(file_pairs)} 个文件")
                        if extract_audio_path or extract_silent_video_path:
                            self._extract_audio_and_silent_video(
                                output_video,
                                extract_audio_path,
                                extract_silent_video_path
                            )
                    else:
                        raise RuntimeError("最终视频文件生成失败")

                except Exception as e:
                    logger.error(f"视频处理过程中出错: {str(e)}")
                    # 尝试保存临时文件以供调试
                    debug_dir = os.path.join(os.path.dirname(output_video), "debug")
                    os.makedirs(debug_dir, exist_ok=True)
                    for f in [temp_main_video, temp_aspect_video, final_output]:
                        if os.path.exists(f):
                            shutil.copy(f, debug_dir)
                    raise RuntimeError(f"视频处理失败，临时文件已保存到 {debug_dir}")

            except Exception as e:
                raise RuntimeError(f"最终合并失败: {str(e)}")




from moviepy.audio.AudioClip import CompositeAudioClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    afx
)
from moviepy import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    afx
)
import traceback
from moviepy.video.tools.subtitles import SubtitlesClip

# 使用示例
if __name__ == "__main__":
    generator = AdvancedSlideshowGenerator(
        output_quality='high',
        target_aspect="9:16",  # 可选: "9:16" 或 "16:9"
        outro_video_path=r"D:\newbegin\NarratoAI\storage\temp\9e27e2c1e41bcd3b1ddf2b32576de3d7.mp4",  # 可选: 结尾视频路径
        text_overlay={
            'text': 'AI文案视频 ，无真人肖像输入',
            'font_size': 30,
            'alpha': 0.75,
            'color': (255, 255, 255),
            'pos_x_ratio': 0.5,
            'pos_y_ratio': 0.95,
            'letter_spacing': 5
        },
        sticker_overlay={
            'text': '免费小说',
            'width': 540,  # 梯形下底宽度
            'height': 60,  # 梯形高度（决定上底宽度）
            'text_size': 24,
            'text_color': (0, 0, 0)
        },
        subtitle_settings={
            'font_size': 28,  # 字幕字体大小
            'color': (255, 255, 255),  # 字幕颜色(白色)
            'outline_color': (0, 0, 0),  # 描边颜色(黑色)
            'outline_width': 2,  # 描边宽度
            'pos_x_ratio': 0.5,  # 水平位置(居中)
            'pos_y_ratio': 0.7,  # 垂直位置(屏幕70%高度)
            'alignment': 'center'  # 对齐方式
        }
    )
    # test_frame = np.ones((600, 800, 3), dtype=np.uint8) * 255
    # result = generator._add_sticker_with_text(test_frame)
    # cv2.imwrite('test_output.jpg', result)
    # generator.generate(
    #     image_folder=r"D:\newbegin\NarratoAI\test_resource\img",
    #     audio_folder=r"D:\newbegin\NarratoAI\test_resource\audio",
    #     output_video=r"D:\newbegin\NarratoAI\output_video006.mp4",
    #     max_workers=4,
    #     # extract_audio_path=r"D:\newbegin\NarratoAI\output_audio.mp3",  # 可选
    #     # extract_silent_video_path=r"D:\newbegin\NarratoAI\output_silent.mp4"  # 可选
    # )

    final_video = r"D:\newbegin\NarratoAI\output_video006.mp4"
    final_output_with_bgm = r"D:\newbegin\NarratoAI\output_video007.mp4"
    bgm_path = r"D:\newbegin\NarratoAI\resource\songs\bgm.mp3"  # 背景音乐路径
    bgm_volume = 0.2  # 背景音乐音量，可以根据需要调整

    if bgm_path and os.path.exists(bgm_path):
        try:
            # 使用FFmpeg合并背景音乐
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', final_video,
                '-i', bgm_path,
                '-filter_complex',
                f'[0:a]volume=1.0[a0];[1:a]volume={bgm_volume},adelay=0|0[a1];[a0][a1]amix=inputs=2:duration=longest[a]',
                '-map', '0:v',
                '-map', '[a]',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                final_output_with_bgm
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            if os.path.exists(final_output_with_bgm) and os.path.getsize(final_output_with_bgm) > 0:
                final_video = final_output_with_bgm
                logger.info(f"已成功添加背景音乐，音量: {bgm_volume}")
            else:
                logger.warning("添加背景音乐失败，使用原视频")
        except subprocess.CalledProcessError as e:
            logger.error(f"添加背景音乐失败: {e.stderr}")
            logger.warning("添加背景音乐失败，使用原视频")

