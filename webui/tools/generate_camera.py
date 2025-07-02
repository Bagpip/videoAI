"""
使用OpenAI API，分析字幕文件，返回剧情梗概和爆点
"""
import traceback
from openai import OpenAI, BadRequestError
import os
import json
import streamlit as st
from pyexpat.errors import messages

from app.config import config
import time
import requests
from volcenginesdkarkruntime import Ark


def generate_camera(tr, params) -> dict:
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

        # 初始化客户端
        global client
        video_provider = config.app.get('video_llm_provider', 'seedance').lower()
        video_api_key = config.app.get(f'video_{video_provider}_api_key')
        video_model = config.app.get(f'video_{video_provider}_model_name')
        video_base_url = config.app.get(f'video_{video_provider}_base_url')

        video_clip_duration = st.session_state.get('video_clip_duration')
        video_fps = st.session_state.get('video_fps')
        rt = st.session_state.get('video_aspect')
        rs = st.session_state.get('video_quality')
        client = Ark(
            base_url=video_base_url,
            api_key=video_api_key,
        )
        camera_id = 0
        for chip in story_prompt:
            background = chip['background']
            camera = chip['camera']
            title = chip['title']
            content = chip['content']
            timestamp = chip['timestamp']

            create_result = client.content_generation.tasks.create(
                model=video_model,
                content = [
                    {
                        "type": "text",
                        "text": f"""
                        你现在是一位专业的动画AI，请将根据小说文本描述生成动画内容，文本内容将包含以下几个方面
                        1、环境描写、人物特征、动态场景为： {background} 
                        2、镜头特写，视觉焦点为： {camera} 
                        3、详细的剧情描述为： {content} 
                        为了视频内容符合法律规范，要严格遵守以下内容：
                        1、涉及士兵，军官的服装不能具象化出现，可用同色制服代替；
                        2、台词内容不要涉及时政（类似发展类字眼）；
                        3、女性角色不能出现关键部位裸露，不能出现不雅动作；
                        4、坚决不能出现对未成年人造成伤害的内容，例如：囚禁，伤害，辱骂，殴打，遗弃等；
                        5、避免出现人物角色缺胳膊少腿问题；
                        6、避免出现大面积血腥和凶器扎到人物身上的类似画面；
                        --rs {rs} --rt {rt} --dur {video_clip_duration} --fps {video_fps} --cf false --wm false --seed -1 
                        """
                    }
                ]
            )
            print("----- polling task status -----")
            task_id = create_result.id
            while True:
                get_result = client.content_generation.tasks.get(task_id=task_id)
                status = get_result.status
                if status == "succeeded":
                    video_url = get_result.content.video_url
                    from tqdm import tqdm

                    output_path = os.path.join(parent_parent_dir, 'storage', 'temp', 'output_videos')
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)  # 递归创建目录
                    save_path = os.path.join(output_path, f"generated_video_{camera_id}.mp4")
                    response = requests.get(video_url, stream=True)
                    total_size = int(response.headers.get("content-length", 0))
                    with open(save_path, "wb") as f, tqdm(
                            desc="下载中",
                            total=total_size,
                            unit="B",
                            unit_scale=True
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                    camera_id = camera_id + 1
                    print("----- task succeeded -----")
                    print(get_result)
                    break
                elif status == "failed":
                    print("----- task failed -----")
                    print(f"Error: {get_result.error}")
                    break
                else:
                    print(f"Current status: {status}, Retrying after 3 seconds...")
                    time.sleep(3)



    except Exception as e:
        raise Exception(f"分析字幕时发生错误：{str(e)}\n{traceback.format_exc()}")

#
# client = Ark(
#     # 此为默认路径，您可根据业务所在地域进行配置
#     base_url="https://ark.cn-beijing.volces.com/api/v3",
#     # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
#     api_key="a95e67a1-6d97-4f17-ac9d-eee1fc3a5323",
# )
# if __name__ == "__main__":
#     print("----- create request -----")
#     # current_dir = os.path.dirname(os.path.abspath(__file__))
#     # parent_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
#     # json_path = os.path.join(parent_parent_dir, 'resource', 'scripts', 'video_story.json')
#     # with open(json_path, 'r', encoding='utf-8') as f:
#     #     data = json.load(f)  # 返回字典或列表
#     # story_prompt = data['plot_titles']
#     messages = [
#         {
#              "type": "text",
#              "text": """你现在是一位专业影视化AI，请将根据文本描述生成电影内容。
#                      文本描述内容为：环保局大楼内，凌志远独自一人检查门窗，环境昏暗，人物疲惫且愤怒。
#                      --resolution 480p  --duration 3 --camerafixed false"""
#         }
#     ]
#     create_result = client.content_generation.tasks.create(
#         model="doubao-seedance-1-0-pro-250528",
#         content= messages,
#     )
#     print(create_result)
#
#     # 轮询查询部分
#     print("----- polling task status -----")
#     task_id = create_result.id
#     while True:
#         get_result = client.content_generation.tasks.get(task_id=task_id)
#         status = get_result.status
#         if status == "succeeded":
#             video_url = get_result.content.video_url
#             from tqdm import tqdm
#
#             folder_name = "output_videos"
#             if not os.path.exists(folder_name):
#                 os.mkdir(folder_name)  # 创建单层目录
#                 current_dir = os.getcwd()
#                 new_folder = os.path.join(current_dir, "output_videos")
#                 save_path = os.path.join(new_folder, "generated_video.mp4")
#             else:
#                 current_dir = os.getcwd()
#                 new_folder = os.path.join(current_dir, "output_videos")
#                 save_path = os.path.join(new_folder, "generated_video.mp4")
#             response = requests.get(video_url, stream=True)
#             total_size = int(response.headers.get("content-length", 0))
#             with open(save_path, "wb") as f, tqdm(
#                     desc="下载中",
#                     total=total_size,
#                     unit="B",
#                     unit_scale=True
#             ) as pbar:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#                     pbar.update(len(chunk))
#
#             print("----- task succeeded -----")
#             print(get_result)
#             break
#         elif status == "failed":
#             print("----- task failed -----")
#             print(f"Error: {get_result.error}")
#             break
#         else:
#             print(f"Current status: {status}, Retrying after 3 seconds...")
#             time.sleep(3)
