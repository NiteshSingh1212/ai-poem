import streamlit as st
import google.generativeai as genai
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure APIs silently
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    openai_client = None

try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Error initializing Gemini client: {e}")
    gemini_model = None

def generate_with_gpt(prompt, style):
    """Try to generate poem with GPT-3.5"""
    if not openai_client:
        return None
    
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a poetry generation AI. Generate a poem in the specified style. Only return the poem text, without any additional commentary or formatting."},
                {"role": "user", "content": f"Write a {style} about {prompt}. Return only the poem text."}
            ],
            max_tokens=300,
            temperature=0.7
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT generation error: {e}")
        return None

def generate_with_gemini(prompt, style):
    """Try to generate poem with Gemini as fallback"""
    if not gemini_model:
        return None
    
    try:
        response = gemini_model.generate_content(
            f"Write a {style} about {prompt}. Return only the poem text, without any additional commentary or formatting."
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        return None

def generate_poem(prompt, style):
    """Main poem generation function with fallback logic"""
    # Try GPT first
    poem = generate_with_gpt(prompt, style)
    if poem:
        logger.info("Successfully generated poem with GPT")
        return poem
    
    # If GPT fails, try Gemini
    logger.info("GPT failed, trying Gemini")
    poem = generate_with_gemini(prompt, style)
    if poem:
        logger.info("Successfully generated poem with Gemini")
        return poem
    
    # If both fail
    raise Exception("Unable to generate poem at this time. Please try again later.")

# Configure page
st.set_page_config(
    page_title="AI Poet",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main-header {
        text-align: center;
        color: #1E3D59;
        padding: 1.5rem 0;
        border-bottom: 2px solid #1E3D59;
        margin: 0 auto 2rem auto;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        max-width: 800px;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #1E3D59 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.2 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 1rem !important;
        width: 100% !important;
    }
    .title-icon {
        font-size: 2rem !important;
        margin-right: 0.5rem !important;
    }
    .title-text {
        font-family: 'Georgia', serif !important;
    }
    .subtitle {
        font-size: 1.2rem !important;
        color: #333 !important;
        font-weight: 500 !important;
    }
    .input-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    .poem-output {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        font-size: 1.1rem;
        color: #1E3D59;
    }
    .poem-output h3 {
        color: #1E3D59;
        font-weight: 600;
        margin-bottom: 1.5rem;
        font-size: 1.4rem;
    }
    .poem-text {
        white-space: pre-line;
        color: #2C3E50;
        font-size: 1.2rem;
        line-height: 1.8;
        font-style: normal;
    }
    .loading-text {
        color: #1E3D59;
        font-size: 1.2rem;
        font-weight: 500;
    }
    .stSpinner {
        text-align: center !important;
        color : black !important;
    }
    .stSpinner > div {
        border-top-color: #1E3D59 !important;
        margin: 0 auto !important;
    }
    .stSpinner > div + div {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #1E3D59 !important;
        background: rgba(255, 255, 255, 0.9) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        margin: 1rem auto !important;
        max-width: 500px !important;
    }
    .stButton > button {
        background-color: #1E3D59 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        font-size: 1.1rem !important;
    }
    .stSelectbox label, .stTextInput label {
        color: #1E3D59 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    .creator-section {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 3rem;
        text-align: center;
    }
    .creator-title {
        font-size: 1.5rem !important;
        color: #1E3D59 !important;
        margin-bottom: 1rem !important;
        font-weight: 600 !important;
    }
    .social-links {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 1rem 0;
    }
    .social-links a {
        transition: transform 0.2s;
    }
    .social-links a:hover {
        transform: translateY(-2px);
    }
    .footer-text {
        color: #666;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown("""
    <h1 class="main-title">
        <span class="title-icon">🖋️</span>
        <span class="title-text">AI Poetry Generator</span>
    </h1>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main content
st.markdown('<div class="input-section">', unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])

with col1:
    theme = st.text_input(
        "🎯 Enter your poem theme:",
        placeholder="e.g., sunset, love, nature..."
    )

with col2:
    style = st.selectbox(
        "📝 Choose poetry style:",
        ["Free verse", "Sonnet", "Haiku", "Limerick"],
        help="Select the style of poetry you want to generate"
    )

if st.button("✨ Generate Poem", type="primary", use_container_width=True):
    if not theme:
        st.warning("🎯 Please enter a theme for your poem!")
    else:
        try:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                with st.spinner("✨ Our AI poet is crafting your masterpiece..."):
                    poem = generate_poem(theme, style)
                    if poem:
                        st.markdown('<div class="poem-output">', unsafe_allow_html=True)
                        st.markdown('<h3 style="color: #1E3D59; font-weight: 600; margin-bottom: 1.5rem; font-size: 1.4rem;">📜 Your Generated Poem</h3>', unsafe_allow_html=True)
                        st.markdown(f'<div class="poem-text">{poem}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error("❌ An error occurred while generating the poem. Please try again.")
st.markdown('</div>', unsafe_allow_html=True)

# Creator section at the bottom
st.markdown("""
    <div class="creator-section">
        <h2 class="creator-title">Created by Nitesh Singh</h2>
        <div class="social-links">
            <a href="https://github.com/NiteshSingh1212" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
            </a>
            <a href="https://www.linkedin.com/in/1212niteshsingh" target="_blank">
                <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
            </a>
        </div>
        <p class="footer-text">This project uses a fine-tuned GPT-2 model to generate unique poetry.<br> 2025 Nitesh Singh. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
