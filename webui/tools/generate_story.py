"""
使用OpenAI API，分析字幕文件，返回剧情梗概和爆点
"""
import traceback
from openai import OpenAI, BadRequestError
import os
import json
from app.config import config
import streamlit as st


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
        video_style = st.session_state.get('video_style', 24)

        client = OpenAI(
            base_url=text_base_url,
            api_key=text_api_key,
        )
        messages = [
            {
                "role": "system",
                "content": f"""
                       你现在是一位专业的小说内容分析师，请按照以下要求处理章节内容：
                       1. **分段规则**：
                        - 以场景转换、时间跳跃或核心事件变化为分段依据,视频故事的开头要有爆点，有吸引力,但结尾留悬念
                        - 每段保证叙事的完整性
                        - 特殊场景（如重要对话/战斗）可独立成段
                        
                        2. **段落结构**：
                        【场景描写】
                        （用现在时态描写可视觉化的元素，包含：）
                        - 环境特征：时间/地点/光影/天气
                        - 角色状态：服装/动作/表情
                        - 关键物体：显著道具/符号化物品
                        - 场景描写用于文本生成图片的提示词，避免出现宗教等敏感词汇导致模型生成图片失败

                        
                        【解说字幕】
                        （用过去时态总结段落核心信息，包含：）
                        - 情节推进：本段发生的关键事件
                        - 情绪基调：紧张/温馨/悬疑等
                        - 隐含线索：需要特别注意的细节
                        （示例："艾琳在婚礼现场发现了丈夫的秘密，血色夕阳预示了悲剧的开始"）
                        
                        3. **输出格式**：
                        请返回一个JSON对象，为每段内容生成包含一个名为"plot_titles"的数组，数组中包含多个对象，每个对象都要包含以下字段：
                       {
                           "plot_titles": [
                               {
                                   "background": "场景描写",
                                   "background_en": "场景描写的英文翻译，确保没有敏感词汇，并加入提示词强调图片风格为{video_style}",
                                   "title": "分析当前场景剧情,给出剧情主题",
                                   "content": "解说字幕",
                                   "timestamp": "解说时长，给出具体时间段，格式为xx:xx:xx,xxx-xx:xx:xx,xxx"
                               }
                           ]
                       }
                        4.请确保返回的是合法的JSON格式。 
                       """  # % (story_clips_num, story_clips_num)
            },
            {
                "role": "user",
                "content": f"小说内容如下：{story_prompt}"
            }
        ]
        # messages = [
        #     {
        #         "role": "system",
        #         "content": """
        #         你现在是一位专业的小说内容分析师，你能根据小说内容生成视频，其中包含了多个5-10秒的电影分镜，请严格执行以下要求：
        #         1、深度解析提供的文学内容，识别核心情节、关键场景和人物关系；
        #         2、提取视觉化元素：包括环境描写、人物特征、动态场景；
        #         3、生成分镜大纲，包含：场景时长建议、镜头类型（特写/全景等）、视觉焦点
        #         4、请返回一个JSON对象，为每个分镜生成包含一个名为"plot_titles"的数组，数组中包含多个对象，每个对象都要包含以下字段：
        #         {
        #             "plot_titles": [
        #                 {
        #                     "background": "当前分镜的环境描写、人物特征、动态场景",
        #                     "camera": "镜头特写，视觉焦点",
        #                     "title": "分析当前场景剧情,给出剧情主题",
        #                     "content": "根据小说内容，以写小说故事的形式给出剧情讲解",
        #                     "timestamp": "分镜时长，给出具体时间段，格式为xx:xx:xx,xxx-xx:xx:xx,xxx"
        #                 }
        #             ]
        #         }
        #         5、请确保返回的是合法的JSON格式。
        #         """ # % (story_clips_num, story_clips_num)
        #     },
        #     {
        #         "role": "user",
        #         "content": f"小说内容如下：{story_prompt}"
        #     }
        # ]
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

