import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/query"

def ask_bot(message, history):
    payload = {"query": message, "k": 4}
    resp = requests.post(API_URL, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        return data["answer"]
    else:
        return f"Error: {resp.status_code}"

gr.ChatInterface(fn=ask_bot).launch()
