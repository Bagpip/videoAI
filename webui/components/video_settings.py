import streamlit as st
from app.models.schema import VideoClipParams, VideoAspect
from webui.tools.generate_camera import generate_camera


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
                    generate_camera(tr, params, )

            with button_cols[1]:
                if st.button(tr("合成视频"), key="save_script", use_container_width=True):
                    print(tr("<UNK>"))

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

    # 视频风格
    video_styles = [
        ("写实", "写实"),
        ("动画", "动画"),
    ]
    video_style = st.selectbox(
        "视频风格",
        options=range(len(video_styles)),
        format_func=lambda x: video_styles[x][0],
        index=0
    )

    # 视频时长
    params.video_clip_duration = st.slider(
        "分镜时长",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="调整每段生成的视频时长"
    )
    video_fps= [
        ("24", 24),
        ("12", 12),
    ]
    params.video_fps = st.selectbox(
        "视频帧率",
        options=range(2),
        format_func=lambda x: video_fps[x][0],
        index=0  # 默认选择 24
    )
    st.session_state['video_aspect'] = params.video_aspect.value
    st.session_state['video_quality'] = video_qualities[quality_index][1]
    st.session_state['video_clip_duration'] = params.video_clip_duration
    st.session_state['video_fps'] = params.video_fps
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
