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


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def make_caption(filestem: str, num_captions: int) -> List[str]:
    client = load_client()

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a politically conscious sound designer. You are tasked with creating a list of comma-separated tags for a given audio file, so we can catalog our sound library. The only information about the audio available to you is the filename. Use the filename to generate tags that describe the audio content. You must write a total of {num_captions} captions.  DO NOT include redundant tags similar to ('sound effect', 'sound art', 'multimedia sound', 'computer music', 'sound design', 'video game audio', 'sound installation') or tags that don't relate to the recorded sound in question. Only include text which is relevant to the sound object or event present in the audio file. For tag ideas, refer to the Universal Category System (UCS). Do not use the term 'ethnic' to refer to non-western sounds or instruments."},
            {"role": "user", "content": f"{filestem}"},
        ], 
    )

    cnt =  completion.choices[0].message.content
    return cnt.split(',')

# TODO: we can only do this 50k at a time, 
# so we can probably run this in a loop. 
MAX_REQUESTS = 1000
MODEL = "gpt-3.5-turbo"

def preprocess(
    dataset: str = vampnet.DATASET, 
    num_captions: int = vampnet.NUM_CAPTIONS,
):
    """
    encodes the audio files with the codec model
    and populates them into the database. 
    """
    assert dataset is not None

    # connect to our datasets table
    conn = vampnet.db.cursor()
    # begin a transatcion
    conn.execute("BEGIN TRANSACTION")

    # get the dataset id and root
    dataset_id, root = conn.execute(f"""
        SELECT id, root 
        FROM dataset 
        WHERE name = '{dataset}'
    """).fetchone()
    print(f"Found dataset {dataset} at {root}")

    # get the audio files that don't have a caption associated with them
    audio_files = conn.execute(f"""
        SELECT id, path
        FROM audio_file
        WHERE dataset_id = {dataset_id}
        AND id NOT IN (
            SELECT audio_file_id
            FROM caption
        )
    """).fetchall()

    audio_files = conn.execute(f"""
        SELECT id, path 
        FROM audio_file
        WHERE dataset_id = {dataset_id}
    """).fetchall()
    print(f"Found {len(audio_files)} audio files")

    tasks = []
    for audio_id, path in tqdm.tqdm(audio_files):
        task = {
            "custom_id": f"task-{audio_id}",
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": f"You are a gen-z, politically conscious sound designer for sound art installation, computer music, film soundtrack and video game audio. You are tasked with creating a list of comma-separated tags) for a given audio file, so we can catalog our sound library. The only information about the audio available to you is the filename. Use the filename to generate tags that describe the audio content. You must write a total of {num_captions} captions.  DO NOT include redundant tags similar to ('sound effect', 'sound art', 'multimedia sound', 'computer music', 'sound design', 'video game audio', 'sound installation') or tags that don't relate to the recorded sound in question. Only include text which is relevant to the sound object or event present in the audio file. For tag ideas, refer to the Universal Category System (UCS). Do not use the term 'ethnic' to refer to non-western sounds or instruments."},
                    {"role": "user", "content": f"{path}"},
                ], 
            }
        }
        if len(tasks) >= MAX_REQUESTS:
            break
        tasks.append(task)

    import json
    task_filename = f"tasks-{dataset}.jsonl"
    with open(task_filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    client = load_client()
    
    batch_file = client.files.create(
        file=open(task_filename, "rb"),
        purpose="batch"
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions", 
        completion_window="24h"
    )


    batch_job = client.batches.retrieve(batch_job.id)
    while batch_job.status != "completed":
        batch_job = client.batches.retrieve(batch_job.id)
        print(batch_job)
        time.sleep(1)

    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content

    with open(result_file_name, 'wb') as file:
        file.write(result)

    result_file_name = "data/batch_job_results.jsonl"

    # Loading data from saved file
    results = []
    with open(result_file_name, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)

    for res in results:
        audio_id = res["custom_id"].split('-')[-1]
        caption_txt_list = res["choices"][0]["message"]["content"].split(',')

        for caption_txt in caption_txt_list:
            caption = vampnet.db.Caption(
                text=caption_txt,
                audio_file_id=audio_id,
            )
            vampnet.db.insert_caption(conn, caption)

    # ask if we should commit
    print(f"Processed {len(audio_files)} audio files")
    # print(f"of which {num_failed} failed")
    
    print("committing changes to the db.")
    breakpoint()
    conn.execute("COMMIT")



if __name__ == "__main__":
    import yapecs
    parser = yapecs.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=vampnet.DATASET, help="dataset to preprocess")

    args = parser.parse_args()
    preprocess(**vars(args))
