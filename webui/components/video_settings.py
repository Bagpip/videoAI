import streamlit as st
from app.models.schema import VideoClipParams, VideoAspect
from webui.tools.generate_camera import generate_camera
from webui.tools.generate_img import generate_img, generate_single_img
from app.config import config

'''
def render_video_panel(tr):
    """渲染视频配置面板"""
    if st.session_state.get('generate_video_setting'):
        with st.container(border=True):
            st.write("生成视频设置")
            params = VideoClipParams()
            render_generate_video_config(params)
            button_cols = st.columns(2)
            with button_cols[0]:
                if st.button(tr("生成视频片段"), key="generate_camera_bt", use_container_width=True):
                    # generate_camera(tr, params, )
                    image_info_list = generate_img(tr, params)
                    if image_info_list:
                        st.success("图片生成成功！")
                        with st.expander("查看生成的图片及提示词", expanded=True):
                            for img_info in image_info_list:
                                st.subheader(f"图片 {img_info['index']}")
                                st.image(img_info['path'], use_column_width=True)
                                st.caption("提示词：")
                                st.text(img_info['prompt'])
                                st.divider()
                    else:
                        st.warning("没有生成任何图片")


    else:
        with st.container(border=True):
            st.write(tr("Video Settings"))
            params = VideoClipParams()
            render_video_config(tr, params)
'''


def render_video_panel(tr):
    """渲染视频配置面板"""
    if st.session_state.get('generate_video_setting'):
        with st.container(border=True):
            st.write("生成视频设置")
            params = VideoClipParams()
            render_generate_video_config(params)
            button_cols = st.columns(2)
            with button_cols[0]:
                if st.button(tr("生成视频片段"), key="generate_camera_bt", use_container_width=True):
                    # 调用生成图片函数并获取结果
                    image_info_list = generate_img(tr, params)
                    # 将图片信息保存到session state中
                    st.session_state['generated_images'] = image_info_list

            # 如果有生成的图片，则展示
            if 'generated_images' in st.session_state and st.session_state.generated_images:
                st.success("图片生成成功！")
                with st.expander("查看生成的图片及提示词", expanded=True):
                    for i, img_info in enumerate(st.session_state.generated_images):
                        # 为每个图片创建编辑区域
                        with st.container(border=True):
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.image(img_info['path'], use_column_width=True)
                            with cols[1]:
                                st.subheader(f"图片 {img_info['index']}")

                                # 使用text_area允许编辑提示词
                                new_prompt = st.text_area(
                                    "提示词",
                                    value=img_info['prompt'],
                                    key=f"prompt_edit_{i}",
                                    height=100
                                )

                                # 如果提示词被修改，更新session state
                                if new_prompt != img_info['prompt']:
                                    st.session_state.generated_images[i]['prompt'] = new_prompt
                                    st.session_state.generated_images[i]['modified'] = True

                                # 显示修改状态
                                if st.session_state.generated_images[i].get('modified', False):
                                    st.warning("提示词已修改但未重新生成")

                                # 重新生成按钮
                                if st.button(
                                        "重新生成此图片",
                                        key=f"regenerate_{i}",
                                        use_container_width=True
                                ):
                                    with st.spinner(f"正在重新生成图片 {img_info['index']}..."):
                                        pixl = st.session_state.get('video_quality')
                                        task_id = st.session_state.get('task_id')
                                        new_img = generate_single_img(
                                            prompt=new_prompt,
                                            size=pixl,
                                            index=img_info['index'],
                                            task_id=task_id
                                        )
                                        if new_img:
                                            st.session_state.generated_images[i] = new_img
                                            st.rerun()

                                st.divider()
    else:
        with st.container(border=True):
            st.write(tr("Video Settings"))
            params = VideoClipParams()
            render_video_config(tr, params)

def render_video_config(tr, params):
    """渲染视频配置"""
    # 视频比例
    video_aspect_ratios = [
        (tr("Portrait"), VideoAspect.portrait.value),
        (tr("Landscape"), VideoAspect.landscape.value),
    ]
    selected_index = st.selectbox(
        tr("Video Ratio"),
        options=range(len(video_aspect_ratios)),
        format_func=lambda x: video_aspect_ratios[x][0],
    )
    params.video_aspect = VideoAspect(video_aspect_ratios[selected_index][1])
    st.session_state['video_aspect'] = params.video_aspect.value

    # 视频画质
    video_qualities = [
        ("4K (2160p)", "2160p"),
        ("2K (1440p)", "1440p"),
        ("Full HD (1080p)", "1080p"),
        ("HD (720p)", "720p"),
        ("SD (480p)", "480p"),
    ]
    quality_index = st.selectbox(
        tr("Video Quality"),
        options=range(len(video_qualities)),
        format_func=lambda x: video_qualities[x][0],
        index=2  # 默认选择 1080p
    )
    st.session_state['video_quality'] = video_qualities[quality_index][1]

    # 原声音量
    params.original_volume = st.slider(
        tr("Original Volume"),
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help=tr("Adjust the volume of the original audio")
    )
    st.session_state['original_volume'] = params.original_volume


def render_generate_video_config(params):
    """渲染视频配置"""
    # 视频比例
    video_aspect_ratios = [
        ("竖屏 9:16（抖音视频）", VideoAspect.portrait.value),
        ("横屏 16:9（西瓜视频）", VideoAspect.landscape.value),
    ]
    selected_index = st.selectbox(
        "视频比例",
        options=range(len(video_aspect_ratios)),
        format_func=lambda x: video_aspect_ratios[x][0],
    )
    params.video_aspect = VideoAspect(video_aspect_ratios[selected_index][1])

    # 视频画质
    # 默认输出分辨率为1024x1024，建议输出分辨率为：
    # · 适用头像： ["768x768", "1024x1024", "1536x1536", "2048x2048"]
    # · 适用文章配图 ：["1024x768", "2048x1536"]
    # · 适用海报传单：["768x1024", "1536x2048"]
    # · 适用电脑壁纸：["1024x576", "2048x1152"]
    # · 适用海报传单：["576x1024", "1152x2048"]

    if config.app.get("video_llm_provider", "flux")=="flux" or config.app.get("video_llm_provider", "flux")=="irag":
        video_qualities = [
            ("16:9推荐", "1024x768"),
            ("9:16推荐", "768x1024"),
            ("适用头像768x768", "768x768"),
            ("适用海报传单2048x2048", "2048x2048"),
            ("适用电脑壁纸2048x1152", "2048x1152"),
        ]
        quality_index = st.selectbox(
            "视频画质",
            options=range(len(video_qualities)),
            format_func=lambda x: video_qualities[x][0],
            index=0  # 默认选择 1080p
        )
        st.session_state['video_quality'] = video_qualities[quality_index][1]

    else:
        video_qualities = [
            ("Full HD (1080p)", "1080p"),
            ("SD (480p)", "480p"),
        ]
        quality_index = st.selectbox(
            "视频画质",
            options=range(len(video_qualities)),
            format_func=lambda x: video_qualities[x][0],
            index=0  # 默认选择 1080p
        )
        st.session_state['video_quality'] = video_qualities[quality_index][1]


    # 视频风格
    video_styles = [
        ("治愈动漫", "治愈动漫"),
        ("仙侠古风", "仙侠古风"),
    ]
    video_style = st.selectbox(
        "视频风格",
        options=range(len(video_styles)),
        format_func=lambda x: video_styles[x][0],
        index=0
    )

    # 视频时长
    # params.video_clip_duration = st.slider(
    #     "分镜时长",
    #     min_value=3,
    #     max_value=10,
    #     value=5,
    #     step=1,
    #     help="调整每段生成的视频时长"
    # )
    # video_fps= [
    #     ("24", 24),
    #     ("12", 12),
    # ]
    # params.video_fps = st.selectbox(
    #     "视频帧率",
    #     options=range(2),
    #     format_func=lambda x: video_fps[x][0],
    #     index=0  # 默认选择 24
    # )
    st.session_state['video_aspect'] = params.video_aspect.value

    # st.session_state['video_clip_duration'] = params.video_clip_duration
    # st.session_state['video_fps'] = params.video_fps
    st.session_state['video_style'] = video_styles[video_style][1]

def get_video_params():
    """获取视频参数"""
    if st.session_state.get('generate_video_setting'):
        return {
            'video_aspect': st.session_state.get('video_aspect', VideoAspect.portrait.value),
            'video_quality': st.session_state.get('video_quality', '1080p'),
            'video_clip_duration': st.session_state.get('video_clip_duration', 5),
            'video_fps': st.session_state.get('video_fps', 24)
        }

    else:
        return {
            'video_aspect': st.session_state.get('video_aspect', VideoAspect.portrait.value),
            'video_quality': st.session_state.get('video_quality', '1080p'),
            'original_volume': st.session_state.get('original_volume', 0.7)
        }


# params = VideoClipParams()
# render_generate_video_config(params)
# video_clip_duration = st.session_state.get('video_clip_duration')
# video_fps = st.session_state.get('video_fps')
# rt = st.session_state.get('video_aspect')
# rs = st.session_state.get('video_quality')
# print("***")
