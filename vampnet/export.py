from pathlib import Path

import argbind

from vampnet.train import VampNetTrainer


def list_models(author: str = "hugggof"):
    from huggingface_hub import HfApi

    api = HfApi()
    models = api.list_models(author=author)
    models = list(models)
    # filter out non-vampnet models
    models = [model for model in models if "vampnetv2" in model.id[len(author)+1:]]
    # sort by last modified
    models = sorted(models, key=lambda x: x.last_modified if x.last_modified is not None else x.created_at, reverse=True)
    # get the ids
    ids = [model.id for model in models]

    return ids


def export_model(
    ckpt: str = None, 
    hf_repo: str = "hugggof/vampnetv2",
    version_tag: str = "latest",
):
    bundle = VampNetTrainer.load_from_checkpoint(ckpt) 

    print("~"*80)
    print(f"loaded checkpoint {ckpt}!")
    print(f"model tag is {bundle.tag}")
    print("~"*80)

    # export the model
    export_repoid = f"{hf_repo}-{bundle.tag}-{version_tag}"
    bundle.push_to_hub(export_repoid)

    print(f"exported model to {export_repoid}!")
    print("testing...")

    # load the model
    bundle = VampNetTrainer.from_pretrained(export_repoid)

    print("done!!!")
    print("you can now load the model pretrained model with the following code:\n")
    print(f"bundle = VampNetTrainer.from_pretrained({export_repoid})")

def run(stage: str = "export"):
    if stage == "export":
        export_model()
    elif stage == "list":
        models = list_models()
        print("\n".join(models))
    else:
        raise ValueError(f"invalid selection {sel}. must be 'export' or 'list'")

if __name__ == "__main__":
    export_model = argbind.bind(export_model, without_prefix=True)
    list_models = argbind.bind(list_models, without_prefix=True)
    run = argbind.bind(run, without_prefix=True)

    args = argbind.parse_args()

    with argbind.scope(args):
        run()

