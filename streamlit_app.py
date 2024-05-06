import streamlit as st
import time
from monolith import run_pipeline
from constants import VIDEO_PATH
import asyncio
import markdown
from IPython.display import display, Markdown



def get_video(file):
    st.video(file)

st.title("Product by Team Hacky Paji")
st.subheader("Get Meeting Insights in minutes.")

video_file = st.file_uploader("Choose a video... (Formats: mp4, mov, avi)")

if video_file is not None:
    video_bytes = video_file.read()
    with open(VIDEO_PATH, 'wb') as out_file: # You can choose whatever filename you like
        out_file.write(video_bytes)

    get_video(VIDEO_PATH) 

    if st.button('Submit'):
        with st.spinner('Running the pipeline...'):
            loop = asyncio.get_event_loop()
            blog  = loop.run_until_complete(run_pipeline())
            
            with open('output.md', 'w') as f:
                f.write(blog)
            
            st.download_button(label="Download output.md",
                           data=blog,
                           file_name='insights.md',
                           mime='text/markdown')
            
            # st.markdown(blog) Streamlit renders markdown incorrectly.
    
    