"""
使用OpenAI API，分析字幕文件，返回剧情梗概和爆点
"""
import traceback
from openai import OpenAI, BadRequestError
import os
import json
from app.config import config


def generate_story(tr, params, story_clips_num, story_prompt) -> dict:
    """分析字幕内容，返回完整的分析结果

    Args:
        srt_path (str): SRT字幕文件路径
        api_key (str, optional): 大模型API密钥. Defaults to None.
        model_name (str, optional): 大模型名称. Defaults to "gpt-4o-2024-11-20".
        base_url (str, optional): 大模型API基础URL. Defaults to None.

    Returns:
        dict: 包含剧情梗概和结构化的时间段分析的字典
    """
    try:
        # 加载字幕文件
        story_prompt = story_prompt

        # 初始化客户端
        global client
        text_provider = config.app.get('text_llm_provider', 'deepseek').lower()
        text_api_key = config.app.get(f'text_{text_provider}_api_key')
        text_model = config.app.get(f'text_{text_provider}_model_name')
        text_base_url = config.app.get(f'text_{text_provider}_base_url')
        client = OpenAI(
            base_url=text_base_url,
            api_key=text_api_key,
        )
        messages = [
            {
                "role": "system",
                "content": """
                你现在是一位专业的小说内容分析师，你能根据小说内容生成视频，其中包含了多个5-10秒的电影分镜，请严格执行以下要求：
                1、深度解析提供的文学内容，识别核心情节、关键场景和人物关系；
                2、提取视觉化元素：包括环境描写、人物特征、动态场景；
                3、生成分镜大纲，包含：场景时长建议、镜头类型（特写/全景等）、视觉焦点
                4、请返回一个JSON对象，为每个分镜生成包含一个名为"plot_titles"的数组，数组中包含多个对象，每个对象都要包含以下字段：
                {
                    "plot_titles": [
                        {
                            "background": "当前分镜的环境描写、人物特征、动态场景",
                            "camera": "镜头特写，视觉焦点",
                            "title": "分析当前场景剧情,给出剧情主题",
                            "content": "根据小说内容，以写小说故事的形式给出剧情讲解",
                            "timestamp": "分镜时长，给出具体时间段，格式为xx:xx:xx,xxx-xx:xx:xx,xxx"
                        }
                    ]
                }
                5、请确保返回的是合法的JSON格式。 
                """ # % (story_clips_num, story_clips_num)
            },
            {
                "role": "user",
                "content": f"小说内容如下：{story_prompt}"
            }
        ]
        # DeepSeek R1 和 V3 不支持 response_format=json_object
        try:
            completion = client.chat.completions.create(
                model=text_model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            summary_data = json.loads(completion.choices[0].message.content)
        except BadRequestError as e:
            completion = client.chat.completions.create(
                model=text_model,
                messages=messages
            )
            # 去除 completion 字符串前的 ```json 和 结尾的 ```
            completion = completion.choices[0].message.content.replace("```json", "").replace("```", "")
            summary_data = json.loads(completion)
        except Exception as e:
            raise Exception(f"大模型解析发生错误：{str(e)}\n{traceback.format_exc()}")

        print(json.dumps(summary_data, indent=4, ensure_ascii=False))

        with open("resource/scripts/video_story.json", 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)
        # 合并结果
        return {
            "plot_summary": summary_data,
        }

    except Exception as e:
        raise Exception(f"分析字幕时发生错误：{str(e)}\n{traceback.format_exc()}")


# 你是一名经验丰富的短剧编剧，擅长根据小说内容按照先后顺序分析关键剧情, 并找出 % s
# 个关键片段。
# 6、请返回一个JSON对象，包含以下字段：
# {
#     "summary": "整体剧情梗概",
#     "plot_titles": [
#         "关键剧情1",
#         "关键剧情2",
#         "关键剧情3",
#         "关键剧情4",
#         "关键剧情5",
#         "..."
#     ]
# }
# 7、请确保返回的是合法的JSON格式, 请确保返回的是 % s
# 个片段。
# """ % (story_clips_num, story_clips_num)