import os
import concurrent.futures
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from moviepy.editor import VideoFileClip
import asyncio
import nest_asyncio
import cv2
import pytesseract
import requests
import time
import numpy as np
from tqdm.auto import tqdm
from openai import AsyncOpenAI
from skimage.metrics import structural_similarity as ssim
from hparams import *
from functools import reduce

# VIDEO PROCESSING
def split_video_chunk(video_path, start, end, chunk_filename):
    """
    Function to split a video chunk and save the audio to a file.
    This function is designed to run in a separate process.
    """
    video = VideoFileClip(video_path).subclip(start, end)
    assert os.path.exists(os.path.dirname(chunk_filename)), "Directory does not exist"
    video.audio.write_audiofile(chunk_filename)
    video.close()  # It's important to close the clip to free up resources.


def split_video(video_path, temp_dir, duration=MAX_DURATION_SEC):
    print(f"Starting to split into mp3 files every {duration / 60:.1f} minutes")
    video = VideoFileClip(video_path)
    total_duration = int(video.duration)
    video.close()  # Close the video clip as we'll open it again in separate processes.

    # Calculate the number of chunks
    num_chunks = (total_duration + duration - 1) // duration  # Ceiling division

    # Prepare arguments for each task
    tasks = []
    for i in range(num_chunks):
        start = i * duration
        end = min(start + duration, total_duration)
        chunk_filename = os.path.join(temp_dir, f"chunk_{i}.mp3")
        tasks.append((video_path, start, end, chunk_filename))

    # Execute tasks in parallel using ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_chunk = {executor.submit(split_video_chunk, *task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            task = future_to_chunk[future]
            try:
                future.result()  # You can handle results or exceptions here
            except Exception as exc:
                print(f'{task[-1]} generated an exception: {exc}')
    
# AUDIO PROCESSING

async def generate_subtitle_for_file(openai_client, file, index, duration=MAX_DURATION_SEC):
    print(f"Loading {os.path.basename(file)}")
    start_time = time.time()
    
    with open(file, "rb") as audio_file:
        response = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="srt",
        )
        
    if isinstance(response, str):
        offset = index * MAX_DURATION_SEC * 1000  # DURATION seconds in milliseconds
        adjusted_subtitle = adjust_timing(response, offset)  # Ensure adjust_timing is an async or a regular function
        end_time = time.time()
        total_time = end_time - start_time
        return adjusted_subtitle
    else:
        print(f"Failed to transcribe {file}")
        print("Response:", response)  # Print the API response
        return None

async def generate_subtitles(openai_client, temp_dir, duration=MAX_DURATION_SEC):
    files = [
        os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp3")
    ]
    
    print("Connecting to OpenAI Whisper-1 API to generate subtitle files")
    
    sorted_file_paths = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    tasks = [generate_subtitle_for_file(openai_client, file, index, duration) for index, file in enumerate(sorted_file_paths)]
    
    all_subtitles = await asyncio.gather(*tasks)
    # Filter out None values in case some transcriptions failed
    all_subtitles = [sub for sub in all_subtitles if sub is not None]
    merged_subtitles = merge_subtitles(all_subtitles)
    return merged_subtitles, all_subtitles

def adjust_timing(subtitle, offset):
    new_subtitle = []
    for line in subtitle.split("\n"):
        if "-->" in line:
            start, end = line.split(" --> ")
            new_start = adjust_time(start, offset)
            new_end = adjust_time(end, offset)
            new_subtitle.append(f"{new_start} --> {new_end}")
        else:
            new_subtitle.append(line)
    return "\n".join(new_subtitle)

def adjust_time(time_str, offset):
    hours, minutes, seconds_milliseconds = time_str.split(":")
    seconds, milliseconds = seconds_milliseconds.split(",")

    total_milliseconds = (
        int(milliseconds)
        + (int(seconds) + (int(minutes) + int(hours) * 60) * 60) * 1000
    )
    total_milliseconds += offset

    new_hours = total_milliseconds // 3600000
    total_milliseconds %= 3600000
    new_minutes = total_milliseconds // 60000
    total_milliseconds %= 60000
    new_seconds = total_milliseconds // 1000
    new_milliseconds = total_milliseconds % 1000

    return f"{new_hours:02}:{new_minutes:02}:{new_seconds:02},{new_milliseconds:03}"
    
def merge_subtitles(subtitles):
    combined_subtitles = "\n".join(subtitles)
    return reindex_subtitles(combined_subtitles)

def reindex_subtitles(srt_content):
    lines = srt_content.split("\n")
    new_content = []
    index = 1

    for line in lines:
        if line.isdigit():
            new_content.append(str(index))
            index += 1
        else:
            new_content.append(line)

    return "\n".join(new_content)

# FRAME PROCESSING

def get_relevant_frames(video_path, duration=MAX_DURATION_SEC, seconds_between_frames=SECONDS_BETWEEN_FRAMES, similarity_threshold=SSIM_THRESHOLD):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    max_frames_per_section = int(MAX_DURATION_SEC * fps)
    print(f"FPS: {fps}, Total frames: {total_frames}, Max frames per section: {max_frames_per_section}")
    selected_section_frames = {i: [] for i in range(int(total_frames//max_frames_per_section)+1)}
    prev_frame = None

    
    # Process video
    for frame_no in tqdm(range(0, total_frames, int(fps * seconds_between_frames))):
        # Set the position of the next frame to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for SSIM calculation
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # Calculate SSIM between frames
            s_value = ssim(prev_frame, gray_frame)

            # If SSIM is below threshold, select frame
            if s_value < similarity_threshold:
                selected_section_frames[frame_no//max_frames_per_section].append({"frame": frame,
                                                                                  "frame_index": frame_no})

        # Update previous frame
        prev_frame = gray_frame

    cap.release()
    return selected_section_frames


def calculate_sharpness(image):
    """Calculate the sharpness of an image using the Laplacian operator."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_contrast(image):
    """Calculate the contrast of an image as the standard deviation of pixel intensities."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast

def get_image_text(image):
    text = pytesseract.image_to_string(image)
    return text

def get_best_frames(frames):
    info_frames = []
    for i, f in tqdm(enumerate(frames)):
        sharpness = calculate_sharpness(f)
        contrast = calculate_contrast(f)
        info_frames.append({"sharpness": sharpness,
                            "contrast": contrast,
                            "text": ""})

def get_sectionwise_quality_frames(shortlisted_frames):
    section_shortlisted_frames = {}
    for section, section_frames in tqdm(shortlisted_frames.items()):
        for section_frame in section_frames:
            section_frame['sharpness'] = calculate_sharpness(section_frame['frame'])
            section_frame['contrast'] = calculate_contrast(section_frame['frame'])
            section_frame['text'] = ''
    
        section_mean_sharpness, section_mean_contrast =  np.mean([_['sharpness'] for _ in section_frames]), np.mean([_['contrast'] for _ in section_frames])
        
        for section_frame in section_frames:
            if section_frame["sharpness"] >= section_mean_sharpness and section_frame["contrast"] >= section_mean_contrast:
                text = get_image_text(section_frame['frame'])
                section_frame['text'] = text
                section_frame['len'] = len(text)
        
        section_mean_text_len = np.mean([_['len'] for _ in section_frames if _['text']!=""])
        section_shortlisted_frames[section] = [section_frame for section_frame in section_frames 
                                               if section_frame['contrast']>=section_mean_contrast 
                                               and section_frame['sharpness']>=section_mean_sharpness 
                                               and section_frame['len']>=section_mean_text_len]
    return section_shortlisted_frames

# Others

def upload_image_to_imgbb(image_array, api_key):
    """
    Uploads an image to imgbb and returns the URL.
    Args:
        image_array: The image array (from OpenCV).
        api_key: Your API key for imgbb.
    Returns:
        The URL of the uploaded image or None if the upload failed.
    """
    # Encode image array to JPEG format in memory
    ret, buf = cv2.imencode('.jpg', image_array)
    if not ret:
        print("Could not encode image.")
        return None
    
    # Send the in-memory binary stream to imgbb
    response = requests.post(
        'https://api.imgbb.com/1/upload',
        files={'image': buf.tobytes()},
        data={'key': api_key}
    )
    
    response_data = response.json()
    
    if response.status_code == 200 and 'data' in response_data:
        return response_data['data']['url']
    else:
        print("Failed to upload image:", response_data.get('error', 'Unknown error'))
        return None

def batch_iterator(input_list, batch_size=1):
    """
    A generator that yields batches of the input list.
    
    Args:
        input_list (list): The list to be batched.
        batch_size (int): The size of each batch.

    Yields:
        list: A batch from the input list of the specified batch size.
    """
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

async def analyze_frame_gpt(openai_client,transcript, image_urls):
    response = await openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ANALYZER_PROMPT.format(transcript),
                    },
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    }
                     for i, image_url in enumerate(image_urls)]
                ],
            }
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content

async def analyze_frame(openai_client,transcript, section_frame):
    description = await analyze_frame_gpt(openai_client, transcript, [section_frame['image_url']])
    section_frame['description'] = description

async def generate_final_content(openai_client,total_content):
    response = asyncio.run(openai_client.chat.completions.create(
        model="gpt-4-turbo-preview", 
        messages=[
            {
            "role": "system",
            "content": "Given a transcript and screenshots of a video, generate a very detailed description blog in a markdown format. Appropriate images must be cited, with captions. Generate table of contents at the beginning. "
            },
            {
            "role": "user",
            "content": "Generate a detailed blog for the following: " + "\n\n\n".join(total_content)
            }
        ],
        temperature=0.2,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        ))
    return response.choices[0].message.content