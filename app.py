import streamlit as st
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
