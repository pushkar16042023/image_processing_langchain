from tempfile import NamedTemporaryFile

import streamlit as st
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from tools import ImageCaptionTool, ObjectDetectionTool


tools = [ImageCaptionTool(), ObjectDetectionTool()]

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

llm = ChatOpenAI(
    openai_api_key='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',
    temperature=0,
    model_name="gpt-3.5-turbo"
)

agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conversational_memory,
    early_stopping_method='generate'
)

# Create tabs
tabs = st.sidebar.radio("Navigation", ["Main", "Info"])

if tabs == "Main":
    # set title
    st.markdown('<style>h1{color:lightblue;}</style>', unsafe_allow_html=True)
    st.title('Ask a question about an image')

    
    st.header("Upload an image")

    
    st.sidebar.title("Question and Response")
    user_question = st.sidebar.text_input('Ask a question:', value='', key='question_input')
    response_placeholder = st.sidebar.empty()  
    uploaded_image_placeholder = st.empty()

    
    file = st.file_uploader("", type=["jpeg", "jpg", "png"])
    if file:
        
        uploaded_image_placeholder.image(file, use_column_width=True)

        
        if user_question and user_question.strip():
            with st.spinner(text="In progress..."):
                with NamedTemporaryFile(dir='.') as f:
                    f.write(file.getbuffer())
                    image_path = f.name

                    
                    response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
                    response_placeholder.write(response)
elif tabs == "Info":
    st.markdown('<style>h1{color:lightblue;}</style>', unsafe_allow_html=True)
    st.title("Image Processing with Transformers")
    st.markdown("""
    In this project, two different transformer-based models are utilized for image processing tasks. 
    The first model, Blip-Image-Captioning, is employed to generate descriptive captions for images. 
    It preprocesses the input image using the BlipProcessor and then uses the BlipForConditionalGeneration 
    model to generate a textual description. The output caption is decoded from the model's response.

    The second model, DETR (DEtection TRansformers), is used for object detection within images. 
    It preprocesses the input image using the AutoImageProcessor and then utilizes the DetrForObjectDetection 
    model to detect objects. The model's output is post-processed to obtain information about the detected 
    objects, including their classes, confidence scores, and locations. These models demonstrate the powerful 
    capabilities of transformer models in handling complex image-related tasks with high accuracy and efficiency.

    Additionally, this functionality is integrated into a Streamlit web application to provide a user-friendly 
    interface. Users can upload images and interact with the models to generate captions and detect objects within 
    the images. Streamlit's interactive features allow for seamless integration of the models into the frontend, 
    enabling users to easily access and utilize the image processing capabilities provided by the models.""")
