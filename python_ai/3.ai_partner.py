import streamlit as st
import os
from openai import OpenAI
import datetime
import json
# 设置页面的配置项
st.set_page_config(
    page_title="AI智能伴侣",
    page_icon="🤖",
    # 布局
    layout="wide",
    # 控制的是侧边栏的状态
    initial_sidebar_state="expanded",
    menu_items={}
)

# 大标题
st.title("AI智能伴侣")

# Logo
st.logo("./resources/logo.png")
#创建OpenAI客户端实例
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")
#系统提示词
system_prompt = """
        你叫 %s，现在是用户的真实伴侣，请完全代入伴侣角色。
        规则：
            1. 每次只回1条消息
            2. 禁止任何场景或状态描述性文字
            3. 匹配用户的语言
            4. 回复简短，像微信聊天一样
            5. 有需要的话可以用❤️🌸等emoji表情
            6. 用符合伴侣性格的方式对话
            7. 回复的内容, 要充分体现伴侣的性格特征
        伴侣性格：
            - %s
        你必须严格遵守上述规则来回复用户。
    """
#生成会话标识
def generate_session_id():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 保存会话
def save_session():
    #保存当前会话
    if st.session_state.current_session:
        #构建新会话对象
        session_data = {
            "session_id": st.session_state.current_session,
            "nickname": st.session_state.nickname,
            "personality": st.session_state.personality,
            "messages": st.session_state.messages
        }
        # 保存新会话到文件
        if not os.path.exists("sessions"):
            os.mkdir("sessions")
        with open(f"sessions/{st.session_state.current_session}.json", "w",encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
#加载会话列表
def load_session():
    session_list = []
    if os.path.exists('sessions'):
        file_list = os.listdir('sessions')
        for file in file_list:
            if file.endswith(".json"):
                session_list.append(file[:-5])
    session_list.sort(reverse=True)#按时间降序排序
    return session_list
#加载会话信息
def load_session_info(session_id):
    try:
        if os.path.exists(f"sessions/{session_id}.json"):
        #读取会话数据
            with open(f"sessions/{session_id}.json", "r",encoding="utf-8") as f:
                session_data = json.load(f)
                st.session_state.nickname = session_data["nickname"]
                st.session_state.personality = session_data["personality"]
                st.session_state.messages = session_data["messages"]
                st.session_state.current_session = session_id
    except:
        st.error("加载会话信息失败")
#删除会话
def delete_session(session_name):
    try:
        if os.path.exists(f"sessions/{session_name}.json"):
            os.remove(f"sessions/{session_name}.json")
            st.success(f"会话 {session_name} 已删除")
            if st.session_state.current_session == session_name:
                st.session_state.messages = []
                st.session_state.current_session = generate_session_id()
    except:
        st.error("删除会话失败")
        
#初始化聊天信息
if 'messages' not in st.session_state:
    st.session_state.messages = []
# 初始化用户输入的昵称和性格
if 'nickname' not in st.session_state:
    st.session_state.nickname = "Sanui"
if 'personality' not in st.session_state:
    st.session_state.personality = "一个有着友好性格的伴侣"
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_id()
    
#展示聊天信息
st.text(f"会话名称：{st.session_state.current_session}")
for message in st.session_state.messages:
    st.chat_message(message["role"],avatar=message["avatar"]).write(message["content"])
    # if message["role"] == "user":
    #     st.chat_message("user",avatar="./resources/user.png").write(message["content"])
    # elif message["role"] == "assistant":
    #     st.chat_message("assistant",avatar="./resources/ai.png").write(message["content"])
    


#侧边栏菜单
with st.sidebar:
    #下面的操作都是在侧边栏中进行的
    
    #会话信息
    st.subheader("控制面板")
    #新建会话
    if st.button("新建会话",width='stretch',icon="✏️"):
        # 保存当前会话
        save_session()
        #创建新会话
        if st.session_state.messages:#如果当前会话有消息，才新建会话，否则不新建
            st.session_state.current_session = generate_session_id()
            st.session_state.messages = []
            # 保存新会话
            save_session()
            st.rerun()
    #历史会话
    session_list = load_session()
    for session in session_list:
        col1,col2 = st.columns([4,1])
        with col1:
            # 点击会话名称，切换到该会话
            if st.button(session,width='stretch',icon="📄",type="primary" if session == st.session_state.current_session else "secondary"):
                load_session_info(session)
                st.rerun()
        with col2:
            # 点击删除按钮，删除该会话
            if st.button("",width='stretch',icon="❌",key=f"delete_{session}"):
                delete_session(session)
                st.rerun()
    st.divider()
    #角色信息
    st.subheader("角色信息")
    # 用户输入的昵称
    nickname = st.text_input("昵称",placeholder='请输入昵称',value=st.session_state.nickname)
    if nickname:
        st.session_state.nickname = nickname
    # 用户输入的性格
    personality = st.text_area("性格",placeholder='请输入性格',value=st.session_state.personality)
    if personality:
        st.session_state.personality = personality



#消息输入框
prompt = st.chat_input("请输入您的问题")
if prompt:
    st.chat_message("user",avatar="./resources/user.png").write(prompt)
    # 保存用户输入的问题
    st.session_state.messages.append({"role": "user", "content": prompt,"avatar":"./resources/user.png"})

    #调用大模型
    response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": system_prompt % (st.session_state.nickname, st.session_state.personality)},
        #将messages列表中的消息都添加到模型中，作为上下文
        *st.session_state.messages
    ],
    #改为流式输出
    stream=True
    )
    # 输出模型回复的内容（非流式输出）
    # print('ai回复:', response.choices[0].message.content)
    # st.chat_message("assistant").write(response.choices[0].message.content)

    # 流式输出模型回复的内容
    response_message = st.empty()
    full_response = ""
    for chunk in response:
        #根据apifox的文档，delta.content是模型回复的内容，流式输出需要拼接起来，才能得到完整的回复
        if chunk.choices[0].delta.content is not None:
            context = chunk.choices[0].delta.content
            full_response += context
            response_message.chat_message("assistant",avatar="./resources/ai.png").write(full_response)
    # 保存模型回复
    st.session_state.messages.append({"role": "assistant", "content": full_response,"avatar":"./resources/ai.png"})
    # 保存当前会话
    save_session()




