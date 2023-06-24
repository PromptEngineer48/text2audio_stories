from transformers import pipeline
import requests
import os

os.environ['HUGGINGFACEHUB_API_TOKEN']='YOUR-HUGGINGFACE_API-KEY'
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

os.environ['OPENAI_API_KEY']='YOUR-API-KEY'

import streamlit as st
import tempfile


#Image to Text Generation
def img2text(url):
    image_to_text = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base", max_new_tokens=100)
    text = image_to_text(url)
    
    # print(text[0]["generated_text"])
    return text[0]["generated_text"]

## Text to Story Generation
#####################################################
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def generate_story(scenario):
    template= """
    You are a story teller
    You can generate a short story based on a simple narrative, the story shoule be no more than 100 words:

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(
        input_variables=["scenario"],
        template=template,
    )

    chain = LLMChain(llm=OpenAI(temperature=1), prompt=prompt)

    story = chain.run(scenario)
    # print(story)
    return story

## Story to Speech Generation
##########################################
def text2speech(message):

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {
        "inputs": message
    }
    
    response = requests.post(API_URL, headers=headers, json=payloads)

    with open('audio.mp3', 'wb') as file:
        file.write(response.content)


## Integration with streamlit
def main():
    st.header("Turn _Images_ into Audio :red[Stories]")

    uploaded_file = st.file_uploader("Choose an image..", type='jpg')
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(bytes_data)
            file_path = file.name

        st.image(uploaded_file, caption='Uploaded Image',use_column_width=True)

        scenario = img2text(file_path)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)

        st.audio("audio.mp3")

if __name__ == "__main__":
    main()
