import streamlit as st
import requests
from PIL import Image
import io
import base64
import os
import random
import time
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Multi-Modal AI Image & Chat Assistant", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

def encode_image(image):
    """Convert PIL Image to base64 string with enhanced error handling"""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def get_image_description(image_b64, api_token):
    """Enhanced image description with context extraction"""
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
    
    try:
        response = requests.post(
            API_URL, 
            headers=headers, 
            json={"inputs": image_b64},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result[0].get('generated_text', 'Generic image')
            return description
        return None
    except Exception as e:
        st.warning(f"Image analysis error: {str(e)}")
        return None

def generate_contextual_image(description: str, api_token: str, 
                               context: Optional[str] = None) -> Optional[Image.Image]:
    """Generate image with advanced context-aware prompt engineering"""
    headers = {"Authorization": f"Bearer {api_token}"}
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    
    # Combine description with optional context
    full_prompt = description
    if context:
        full_prompt += f", inspired by context: {context}"
    
    enhanced_prompt = (
        f"Detailed, high-quality image of {full_prompt}. "
        "Professional composition, photorealistic, 4K resolution, "
        "perfect lighting and clarity"
    )
    
    negative_prompt = (
        "low quality, blurry, distorted, ugly, bad proportions, "
        "poorly detailed, unprofessional"
    )
    
    payload = {
        "inputs": enhanced_prompt,
        "parameters": {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": negative_prompt,
            "seed": random.randint(1, 1000000)
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        return Image.open(io.BytesIO(response.content)) if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Image generation error: {str(e)}")
        return None

def generate_variations(image, api_token, num_variations=3):
    """Generate image variations with context preservation"""
    image_b64 = encode_image(image)
    image_description = get_image_description(image_b64, api_token)
    
    if not image_description:
        st.warning("Could not extract image description")
        return []
    
    variations = []
    for _ in range(num_variations):
        variation = generate_contextual_image(
            image_description, 
            api_token, 
            context="creative variation"
        )
        if variation:
            variations.append(variation)
    
    return variations

def main():
    # Environment setup
    HF_TOKEN = os.getenv('HF_AUTH_TOKEN')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    if not all([HF_TOKEN, GROQ_API_KEY]):
        st.error("Missing required API keys")
        return

    st.title("🖼️ Multi-Modal AI Assistant")

    # Tabs for different interactions
    tab1, tab2 = st.tabs(["🎨 Image Generation", "💬 Context Chat"])

    with tab1:
        st.header("Image Generation & Variations")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            try:
                input_image = Image.open(uploaded_file)
                st.image(input_image, width=300, caption="Uploaded Image")
                
                if st.button("🔄 Generate Variations"):
                    variations = generate_variations(input_image, HF_TOKEN)
                    
                    if variations:
                        st.header("Image Variations")
                        cols = st.columns(len(variations))
                        for col, variation in zip(cols, variations):
                            with col:
                                st.image(variation, use_container_width=True)
                    else:
                        st.warning("Could not generate variations")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    with tab2:
        st.header("Contextual Image Chat")
        
        # Optional image upload
        chat_image = st.file_uploader(
            "Upload an optional context image", 
            type=["jpg", "png", "jpeg"], 
            key="context_image"
        )
        
        # Chat LLM
        llm = ChatGroq(
            temperature=0.7,
            model_name="gemma-7b-it",
            groq_api_key=GROQ_API_KEY
        )
        
        # Chat input
        user_input = st.text_input(
            "Enter your message", 
            placeholder="Ask about the image or request image generation..."
        )
        
        if st.button("Send Message"):
            if user_input:
                messages = [
                    SystemMessage(content="You are a helpful multi-modal AI assistant")
                ]
                
                # Process context image if uploaded
                if chat_image:
                    context_img = Image.open(chat_image)
                    img_b64 = encode_image(context_img)
                    img_description = get_image_description(img_b64, HF_TOKEN)
                    
                    if img_description:
                        messages.append(HumanMessage(
                            content=f"Context image description: {img_description}"
                        ))
                
                # Add user input
                messages.append(HumanMessage(content=user_input))
                
                # Generate response
                with st.spinner("Thinking..."):
                    response = llm.invoke(messages).content
                    st.write(response)
                
                # Check for image generation request
                if any(phrase in user_input.lower() for phrase in 
                       ["generate image", "create picture", "draw"]):
                    image_prompt = user_input.split("image")[-1].strip()
                    generated_image = generate_contextual_image(
                        image_prompt, 
                        HF_TOKEN, 
                        context=img_description if chat_image else None
                    )
                    
                    if generated_image:
                        st.header("Generated Image")
                        st.image(generated_image)

if __name__ == "__main__":
    main()