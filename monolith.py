import os
import cv2
import json
import yaml
import time
import shutil
import asyncio
import logging
import requests
import tempfile
import importlib
import numpy as np
import nest_asyncio
from tqdm.auto import tqdm
from openai import AsyncOpenAI
import matplotlib.pyplot as plt 
from functools import reduce,partial

nest_asyncio.apply()

from hparams import *
from constants import *
from utils import (split_video, generate_subtitles, get_relevant_frames, \
                    get_sectionwise_quality_frames, upload_image_to_imgbb, analyze_frame,generate_final_content )


logging.getLogger().setLevel(logging.INFO)


def load_secrets():
    with open('secrets.yaml', 'r') as file:
        secrets = yaml.load(file, Loader=yaml.FullLoader)

    for key, value in secrets.items():
        os.environ[key] = value

def reset_run_directory():
    for filename in os.listdir(BASE_DIR):
        file_path = os.path.join(BASE_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

async def split_video_into_chunks(metadata):
    split_video(VIDEO_PATH, BASE_DIR, duration=MAX_DURATION_SEC)
    logging.info(f"Video Saved into chunks: {BASE_DIR} with max duration as: {MAX_DURATION_SEC}")
    return metadata

async def save_subtitles(metadata):
    subtitles, sectionwise_subtitles = asyncio.run(generate_subtitles(metadata['openai_client'], BASE_DIR, duration=MAX_DURATION_SEC))
    with open(SRT_PATH, "w", encoding="utf-8") as file:
        file.write(subtitles)
    file.close()
    logging.info(f"Subtitles Saved in: {SRT_PATH}")
    metadata['subtitles'] = subtitles
    return metadata

async def shortlist_frames(metadata):
    shortlisted_frames = get_relevant_frames(VIDEO_PATH, duration=MAX_DURATION_SEC, seconds_between_frames=SECONDS_BETWEEN_FRAMES, similarity_threshold=SSIM_THRESHOLD)
    metadata['shortlisted_frames'] = shortlisted_frames
    logging.info(f"Shortlisted: {sum([len(v) for k,v in shortlisted_frames.items()])} after SSIM")
    return metadata

async def sectionwise_shortlist_frames(metadata):
    # sectionwise_shortlisted_frames = get_sectionwise_quality_frames(metadata['shortlisted_frames'])  # 
    # sectionwise_shortlisted_frames_count = sum([len(v) for v in sectionwise_shortlisted_frames.values()])
    # logging.info(f"Total selected frames section wise: {sectionwise_shortlisted_frames_count}")
    # In case of shorter-videos, sometimes mean contrast/sharpness brightness above logic fails.
    sectionwise_shortlisted_frames = metadata['shortlisted_frames'].copy()
    metadata['sectionwise_shortlisted_frames'] = sectionwise_shortlisted_frames
    return metadata

async def save_frames(metadata):
    if not os.path.exists(os.path.join(BASE_DIR,FRAMES_DIR)):  # This checks if the save directory exists, creates if it don't
        os.mkdir(os.path.join(BASE_DIR,FRAMES_DIR))
    SAVE_DIR = os.path.join(BASE_DIR,FRAMES_DIR)
    sectionwise_shortlisted_frames = metadata['sectionwise_shortlisted_frames']
    for section, section_frames in sectionwise_shortlisted_frames.items(): 
        NSHOW_IMAGES = min(MAX_IMAGES, len(section_frames))
        NCOL = 10
        NROW = NSHOW_IMAGES // NCOL + 1
        if NSHOW_IMAGES == 0:
            continue
        
        for i, frame in tqdm(enumerate(section_frames[:NSHOW_IMAGES])):         
            save_path = os.path.join(SAVE_DIR, f"section_{section}_frame_{i}.png")
            cv2.imwrite(save_path, frame['frame']) 
    return metadata

async def upload_frames(metadata):
    sectionwise_shortlisted_frames = metadata['sectionwise_shortlisted_frames']

    for section, section_frames in tqdm(sectionwise_shortlisted_frames.items()):
        for section_frame in section_frames:
            image_url = upload_image_to_imgbb(section_frame['frame'], metadata['IMGBB_API_KEY'])
            if image_url:
                section_frame['image_url'] = image_url
            else:
                logging.error("Got NIL image_url, frames couldn't be uploaded.")
    return metadata


async def analyze_all_frames(metadata):
    """Analyze all frames for each section in parallel."""

    sectionwise_shortlisted_frames = metadata['sectionwise_shortlisted_frames']
    subtitles = metadata['subtitles']
    tasks = []
    for section, section_frames in sectionwise_shortlisted_frames.items():
        for section_frame in section_frames:
            task = analyze_frame(metadata['openai_client'],subtitles[section], section_frame)
            tasks.append(task)
    await asyncio.gather(*tasks)
    return metadata

async def generate_blog(metadata):
    sectionwise_shortlisted_frames = metadata['sectionwise_shortlisted_frames']
    subtitles = metadata['subtitles']
    total_content = []
    for section, section_frames in tqdm(sectionwise_shortlisted_frames.items()):
        lines = subtitles[section].split('\n')
        text_lines = [line for line in lines if not line.isdigit() and '-->' not in line]
        text_content = '\n'.join(text_lines).strip()
        reference = "\n".join([f"{frame['description']}![source]({frame['image_url']})" for frame in section_frames])
        if reference.strip() == "":
            reference = None
        section_content = f"\n\nTranscript:\n{text_content}.\n\nCorresponding References:\n{reference}"
        total_content.append(section_content)
    metadata["total_content"] = total_content
    metadata["blog"] = await generate_final_content(metadata['openai_client'],total_content)

    return metadata

async def run_pipeline():
    load_secrets()
    # reset_run_directory()
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    IMGBB_API_KEY = os.environ["IMGBB_API_KEY"]
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    metadata = {"openai_client":openai_client,"IMGBB_API_KEY":IMGBB_API_KEY}


    pipeline = [split_video_into_chunks,
    save_subtitles, shortlist_frames, sectionwise_shortlist_frames, save_frames, 
    upload_frames,analyze_all_frames, generate_blog]

    for component in pipeline:
            logging.info(f"Starting Component: {component.__name__}")
            metadata = await component(metadata)
            logging.info(f"Completed Component: {component.__name__}")

    return metadata["blog"]
    


