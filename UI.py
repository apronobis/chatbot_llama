import gradio as gr
from chatbot import getAnswer


def chatbot(text):
    # Your chatbot model or logic goes here
    # For this example, we'll just return a simple response
    responses = {
        "hello": "Hi!",
        "how are you": "I'm good, thanks!",
        "what's your name": "I'm a chatbot!",
        "exit": "Goodbye"
    }
    return getAnswer(text)
    # return responses.get(text.lower(), "I didn't understand that.")


demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Type a message"),
    outputs=gr.Textbox(label="Chatbot response"),
    title="Simple Chatbot",
    description="A simple chatbot demo"
)
demo.launch()
