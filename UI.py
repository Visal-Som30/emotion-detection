import streamlit as st
from PIL import Image
import io
import emoji
from FaceDetection import FaceDetection
import numpy as np
import base64

fd = FaceDetection()

def instuction():
    with st.sidebar:
        st.header("📑 Instruction")
        st.markdown("1. Drag and Drop your photo into the right form. 👉")
        st.markdown("2. Just Click on the DETECT button, it will tell the emotion from your photo.")
  
        # Display another hint with a different emoji using st.markdown
        st.markdown(f"{emoji.emojize(':bulb:')} **Note:** Angry, Neutral, and Happy Emotions Only.")

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

        # Read the contents of the file
        image_bytes = uploaded_file.read()

        # Use PIL to open the image from bytes
        image = np.array(Image.open(uploaded_file))
        # st.write(image)
        # print(image)
        result = fd.predict(image)
        if result == "Happy":
            st.write(f"You are {result} 😊 right now! Maybe your crush accepted your confession.")
        elif result == "Angry":
            st.write(f"You are {result} 😡 right now! Maybe your crush is dating with another guy.")
        elif result == "Neutral":
            st.write(f"You are {result} 😐 right now! Maybe your crush is considering your confession.")
        else:
            st.write("🙏 Sorry, we cannot detect your photo. Please Kindly upload another photo. 🙏")
        # Set button_clicked to True to indicate that the button has been clicked
        st.session_state.dislike_button = True

    if uploaded_file is not None:

        # image_bytes = uploaded_file.read()

        # # Use PIL to open the image from bytes
        # image = Image.open(io.BytesIO(image_bytes))

        # # Display the image with a smaller size
        # st.image(image, width=500) 
        image_bytes = uploaded_file.getvalue()

        # Display the uploaded image in the center using markdown
        st.markdown(
            f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(image_bytes).decode()}",width=100px></div>',
            unsafe_allow_html=True,
        )

      
    instuction()

if __name__ == "__main__":
    main()


