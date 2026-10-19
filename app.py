cat > app.py << 'PYEOF'
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="我的智能助手", layout="centered")

st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg, #f5f8fd 0%, #eef2f9 100%); }
  .block-container { padding-top: 2.2rem; }
  h1 {
    background: linear-gradient(90deg, #1d4ed8, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
  }
  [data-testid="stChatMessage"] {
    border-radius: 16px;
    border: 1px solid rgba(15,23,42,0.06);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(29,78,216,0.06);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #ffffff;
  }
  [data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid rgba(15,23,42,0.06); }
  .stButton button { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

@tool
def calculator(expression: str) -> str:
    """可以计算数学表达式，遇到数学运算必须调用这个工具。
    Args:
        expression: 数学表达式字符串，例如 "2+3*5"
    """
    return str(eval(expression))

@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户问现在几点、今天几号时调用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with st.sidebar:
    st.markdown("### 助手信息")
    st.caption("模型：DeepSeek Chat")
    st.caption("框架：LangChain + LangGraph")
    st.caption("工具：计算器 · 时间查询")
    st.caption("记忆：会话级记忆")
    st.divider()
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "agent" not in st.session_state:
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        temperature=0,
    )
    st.session_state.agent = create_agent(
        model=llm,
        tools=[calculator, get_current_time],
        system_prompt="你是一个贴心的生活小助手。记住用户的个人信息，用工具回答计算和时间问题。",
        checkpointer=MemorySaver(),
    )
    st.session_state.config = {"configurable": {"thread_id": "web-assistant"}}

if "messages" not in st.session_state:
    st.session_state.messages = []

def ask(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    res = st.session_state.agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=st.session_state.config,
    )
    used_tools = [m.name for m in res["messages"] if getattr(m, "type", "") == "tool"]
    reply = res["messages"][-1].content
    st.session_state.messages.append({"role": "assistant", "content": reply, "tools": used_tools})

st.title("我的智能助手")
st.caption("会算数 · 会报时间 · 记得你 · 多轮对话")

if st.session_state.get("pending"):
    pending = st.session_state.pending
    st.session_state.pending = None
    with st.spinner("思考中..."):
        ask(pending)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools"):
            st.caption("调用了工具：" + "、".join(msg["tools"]))

if not st.session_state.messages:
    st.markdown("##### 试试这些")
    cols = st.columns(3)
    for col, ex in zip(cols, ["2367*89等于多少？", "现在几点了？", "记住我叫小晨"]):
        if col.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

if prompt := st.chat_input("输入你的问题..."):
    with st.spinner("思考中..."):import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 页面设置
st.set_page_config(page_title="DeepSeek智能助手", layout="wide")
st.title("DeepSeek对话助手")

# 填写DeepSeek密钥
deepseek_api_key = st.sidebar.text_input("DeepSeek密钥", type="password")

# 接入DeepSeek接口
model = ChatOpenAI(
    model="deepseek-chat",
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1"
)

# 对话状态
class ChatState(TypedDict):
    question: str
    answer: str

def chat_node(state: ChatState):
    resp = model.invoke(state["question"])
    return {"answer": resp.content}

# 搭建工作流
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)
app = graph.compile()

# 聊天输入框
msg = st.chat_input("请输入你的问题")
if msg and deepseek_api_key:
    with st.chat_message("user"):
        st.write(msg)
    res = app.invoke({"question": msg})
    with st.chat_message("assistant"):
        st.write(res["answer"])
elif not deepseek_api_key:
    st.info("请在侧边栏填写你的DeepSeek密钥")
