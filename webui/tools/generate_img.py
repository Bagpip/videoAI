# 请安装 OpenAI SDK : pip install openai
# 请安装 requests: pip install requests
from openai import OpenAI
import requests
import traceback
import os
import json
import streamlit as st
from app.config import config
import time
import requests


def generate_img(tr, params) -> dict:
    """分析字幕内容，返回完整的分析结果

    Args:
        srt_path (str): SRT字幕文件路径
        api_key (str, optional): 大模型API密钥. Defaults to None.
        model_name (str, optional): 大模型名称. Defaults to "gpt-4o-2024-11-20".
        base_url (str, optional): 大模型API基础URL. Defaults to None.

    Returns:
        dict: 包含剧情梗概和结构化的时间段分析的字典
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    json_path = os.path.join(parent_parent_dir, 'resource', 'scripts', 'video_story.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 返回字典或列表
    try:
        # 加载字幕文件
        story_prompt = data['plot_titles']
        image_info_list = []  # 存储图片信息和提示词

        # 初始化客户端
        global client
        video_provider = config.app.get('video_llm_provider', 'flux').lower()
        video_api_key = config.app.get(f'video_{video_provider}_api_key')
        video_model = config.app.get(f'video_{video_provider}_model_name')
        video_base_url = config.app.get(f'video_{video_provider}_base_url')

        pixl = st.session_state.get('video_quality')

        task_id = st.session_state.get('task_id')

        client = OpenAI(
            base_url=video_base_url,
            api_key=video_api_key
        )

        camera_id = 0
        for chip in story_prompt:
            if video_provider == 'flux':
                background = chip['background_en']
            else:
                background = chip['background']

            response = client.images.generate(
                model= video_model,
                prompt=f"{background}",
                size=f"{pixl}",
                n=1,
                extra_body={
                    "steps": 4,
                    "guidance": 3.5
                }
            )

            # 正确获取图片URL的方式
            image_url = response.data[0].url

            from tqdm import tqdm

            output_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'output_img', task_id)
            if not os.path.exists(output_path):
                os.makedirs(output_path)  # 递归创建目录
            camera_id = camera_id + 1
            save_path = os.path.join(output_path, f"{camera_id}.png")


            # 发送 HTTP GET 请求获取图片数据
            img_response = requests.get(image_url)  # 使用 requests 库获取图片

            # 检查请求是否成功
            if img_response.status_code == 200:

                # 写入文件
                with open(save_path, "wb") as file:
                    file.write(img_response.content)
                    # 保存图片信息和提示词
                    image_info_list.append({
                        'path': save_path,
                        'prompt': background,
                        'index': camera_id
                    })
            else:
                st.error(f"图片 {camera_id} 下载失败，HTTP 状态码: {img_response.status_code}")

        return image_info_list  # 返回图片信息列表


    except Exception as e:
        raise Exception(f"分析字幕时发生错误：{str(e)}\n{traceback.format_exc()}")


def generate_single_img(prompt: str, size: str, index: int, task_id: str) -> dict:
    """生成单张图片

    Args:
        prompt: 图片提示词
        size: 图片尺寸
        index: 图片序号
        task_id: 任务ID

    Returns:
        dict: 包含图片路径和提示词的信息
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

        # 初始化客户端
        video_provider = config.app.get('video_llm_provider', 'flux').lower()
        video_api_key = config.app.get(f'video_{video_provider}_api_key')
        video_model = config.app.get(f'video_{video_provider}_model_name')
        video_base_url = config.app.get(f'video_{video_provider}_base_url')

        client = OpenAI(
            base_url=video_base_url,
            api_key=video_api_key
        )

        response = client.images.generate(
            model=video_model,
            prompt=prompt,
            size=size,
            n=1,
            extra_body={
                "steps": 4,
                "guidance": 3.5
            }
        )

        image_url = response.data[0].url
        output_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'output_img', task_id)
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, f"{index}.png")

        img_response = requests.get(image_url)
        if img_response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(img_response.content)

            return {
                'path': save_path,
                'prompt': prompt,
                'index': index
            }
        else:
            st.error(f"图片 {index} 下载失败，HTTP 状态码: {img_response.status_code}")
            return None

    except Exception as e:
        st.error(f"生成图片时发生错误：{str(e)}")
        st.text(traceback.format_exc())
        return None