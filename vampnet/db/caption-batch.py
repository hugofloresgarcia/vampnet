""" 
using a language model, create captions for a given audio clip
"""
from typing import List
from pathlib import Path
import audiotools as at

import argbind
import tqdm
import sqlite3
import pandas as pd

import vampnet
import backoff
import openai 
import time

RAISE = True


def load_client():
    if not hasattr(load_client, "_client"):
        import openai
        load_client._client = openai.Client()

    return load_client._client


# TODO: we can only do this 50k at a time, 
# so we can probably run this in a loop. 
MAX_REQUESTS = 1000
MODEL = "gpt-3.5-turbo"

def caption_batch(
    num_captions: int = MAX_REQUESTS,
):
    # a list of audio files
    audio_files: list[at.AudioSignal]

    # make a list of tasks
    tasks = []
    # go thru the audio files
    for audio_id, path in tqdm.tqdm(audio_files):
        task = {
            "custom_id": f"task-{audio_id}", # a unique id
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": MODEL,
                # prompt goes here
                "messages": [
                    {"role": "system", "content": f"You are a gen-z, politically conscious sound designer for sound art installation, computer music, film soundtrack and video game audio. You are tasked with creating a list of comma-separated tags) for a given audio file, so we can catalog our sound library. The only information about the audio available to you is the filename. Use the filename to generate tags that describe the audio content. You must write a total of {num_captions} captions.  DO NOT include redundant tags similar to ('sound effect', 'sound art', 'multimedia sound', 'computer music', 'sound design', 'video game audio', 'sound installation') or tags that don't relate to the recorded sound in question. Only include text which is relevant to the sound object or event present in the audio file. For tag ideas, refer to the Universal Category System (UCS). Do not use the term 'ethnic' to refer to non-western sounds or instruments."},
                    {"role": "user", "content": f"{path}"},
                ], 
            }
        }
        # add the task to the list if we haven't reached the max
        if len(tasks) >= num_captions:
            break
        tasks.append(task)

    # save the tasks to a file
    import json
    task_filename = f"tasks-{dataset}.jsonl"
    with open(task_filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # load the openai client
    client = load_client()
    
    # create the task file on openai
    batch_file = client.files.create(
        file=open(task_filename, "rb"),
        purpose="batch"
    )

    # create a batch job
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions", 
        completion_window="24h"
    )

    # wait for the batch job to complete
    batch_job = client.batches.retrieve(batch_job.id)
    while batch_job.status != "completed":
        batch_job = client.batches.retrieve(batch_job.id)
        print(batch_job)
        time.sleep(1)

    # get the results
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    # save the results to a file
    result_file_name = "data/batch_job_results.jsonl"
    with open(result_file_name, 'wb') as file:
        file.write(result)


    # Loading data from saved file
    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)

    for res in results:
        audio_id = res["custom_id"].split('-')[-1] # that unique id we created earlier
        caption_txt_list = res["choices"][0]["message"]["content"].split(',') # the caption text
        print(f"audio_id: {audio_id}, caption_txt_list: {caption_txt_list}")

    print("all good.")
    breakpoint()
    conn.execute("COMMIT")



if __name__ == "__main__":
    import yapecs
    parser = yapecs.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=vampnet.DATASET, help="dataset to preprocess")

    args = parser.parse_args()
    preprocess(**vars(args))
