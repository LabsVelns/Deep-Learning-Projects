import streamlit as st
import os

from test import process_video_with_esrgan
from SidebySide import create_side_by_side_video

model=r"C:\Users\harsh\Downloads\VID_tune2-2.pth"
output_video_path = r"C:\Users\harsh\Downloads\Project Run\input_vs_output.mp4"

def main():
    #video
    st.header("VIDEO ESR")
    
    videos=st.file_uploader("please Upload ur Video & Download ESR Video",type=["mp4","mkv","avi"])
    if videos is not None:
        
        path_in = videos.name
        st.write(path_in)
        with open(path_in, "wb") as f:
            f.write(videos.read())
        st.success("Video uploaded successfully.")
    
        
        if st.button("Submit"):
            st.write("Button clicked!")
            var = process_video_with_esrgan(videos.name,model)
            
            
            if var is not None:
                st.success("Video Has been Processed")
                base = os.path.splitext(os.path.basename(var))[0]
                path = r'C:\Users\harsh\Downloads\Project Run\{:s}.avi'.format(base)
                st.write('Processing the Output....')
                create_side_by_side_video(videos.name,path,output_video_path)
                st.video(output_video_path)
                
main()          
            
                
                





                # path_in2 = var
                # st.write(path_in2)
                # with open(path_in2, "wb") as f2:
                #     f2.write(var.read())
                    
                
    # if submit == "process_video_with_esrgan":
    #     output=(process_video_with_esrgan(videos,model).decode("utf-8"))
    #     # st.open(output)


# if __name__ == "__main__":
#     main()
# with open(var, "wb") as f:
            #     f.write(videos.read())
            # st.write(var)