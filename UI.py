import streamlit as st
from PIL import Image
import pandas as pd

def instuction():
    with st.sidebar:
        st.header("📑 Instruction")
        st.markdown("1. Drag and Drop your photo into the right form. 👉")
        st.markdown("2. Just Click on the DETECT button, it will tell the emotion from your photo.")

def main():
    st.set_page_config(
        page_title="Emotion Detection",
        page_icon="🇰🇭",
        initial_sidebar_state="expanded",
        menu_items={"About": "Built by @dcarpintero with Streamlit & NewsAPI"},
    )
    
    st.title("😃 Emotion Detection 😔")

    # Upload image through Streamlit's file uploader
    uploaded_file = st.file_uploader("Please Input Your Image Here to Detect Your Emotion 👇", type=["jpg", "jpeg", "png"])
    
    # Initialize session state
    if "detect_button" not in st.session_state:
        st.session_state.detect_button = False

    # Display the button with a unique label
    detect_button = st.button("DETECT")
    
    # Check if the button has not been clicked before
    if detect_button and not st.session_state.detect_button:
        st.write("You are upset 😔 right now! Maybe your crush rejected your confession.")
        # Set button_clicked to True to indicate that the button has been clicked
        st.session_state.dislike_button = True

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

      
    instuction()

if __name__ == "__main__":
    main()


