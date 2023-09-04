import os
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
from moviepy.video.fx.crop import crop

def create_side_by_side_video(video_path1, video_path2, output_path):
    # Define text labels
    file1_txt = os.path.splitext(os.path.basename(video_path1))[0]
    file2_txt = os.path.splitext(os.path.basename(video_path2))[0]

    # Load video clips
    clip1 = VideoFileClip(video_path1)
    clip2 = VideoFileClip(video_path2)

    # Get video dimensions
    width = clip1.size[0] + clip2.size[0]
    height = max(clip1.size[1], clip2.size[1])

    # Resize and crop video clips
    clip1 = clip1.resize(height=height)
    clip2 = clip2.resize(height=height)
    clip1 = crop(clip1, width=clip1.size[0] // 2, x_center=clip1.size[0] // 4)
    clip2 = crop(clip2, width=clip2.size[0] // 2, x_center=clip2.size[0] // 4)

    # Create text clips
    txt_clip1 = TextClip(file1_txt, fontsize=36, color="white")
    txt_clip2 = TextClip(file2_txt, fontsize=36, color="white")

    # Position text clips
    txt_clip1 = txt_clip1.set_position("center").set_duration(clip1.duration)
    txt_clip2 = txt_clip2.set_position((clip1.size[0], 0)).set_duration(clip2.duration)

    # Create the final side-by-side video
    final_clip = clips_array([[clip1, clip2]])
    final_clip = CompositeVideoClip([final_clip, txt_clip1, txt_clip2])

    # Write the output video
    final_clip.write_videofile(output_path, codec="libx264")

# Example usage:
# file1_path = r"C:\Users\harsh\Downloads\Prathamesh\car (240p).mp4"
# file2_path = r"C:\Users\harsh\Downloads\Prathamesh\car (240p)_ESRGAN.avi"
# output_video_path = r"C:\Users\harsh\Downloads\Prathamesh\input_vs_output.mp4"
# create_side_by_side_video(file1_path, file2_path, output_video_path)
