"""
export the model from the last run, 
copying it to models/ to use it in the interface
and optionally uploading it to the huggingface hub.
"""


from huggingface_hub import HfApi

import vampnet


def export_model():
    api = HfApi()
    username = vampnet.HF_USERNAME
    repo_name = vampnet.HF_REPO_NAME
    config = vampnet.CONFIG
    repo_id = vampnet.REPO_ID
    model_tag = vampnet.EXPORT_MODEL_TAG
    output_model_path = vampnet.MODEL_FILE

    # copy the model file
    import shutil
    shutil.copy(vampnet.RUNS_DIR / model_tag / "vampnet" / "weights.pth", output_model_path)
    print(f"Model saved locally to {output_model_path}")


    # create a repo on the hub
    print(f"Creating repo {repo_id} on the hub.")
    api.create_repo(f"{username}/{repo_name}", exist_ok=True)

    # upload the model
    print(f"Uploading model to {repo_id}")
    future = api.upload_file(
        repo_id=repo_id,
        path_in_repo=str(output_model_path.relative_to(vampnet.ROOT)),
        path_or_fileobj=output_model_path,
        run_as_future=True,
    )
    future.result()
    print(f"Model uploaded to {repo_id}. !!!")

    print(f"you may now use the model in the interface by running `vampnet.load_hf_model('{repo_id}:{vampnet.MODEL_FILE.stem}')`")

if __name__ == "__main__":
    export()