from pathlib import Path

import argbind

from scripts.exp.train import VampNetTrainer


@argbind.bind(without_prefix=True)
def export_model(
    ckpt: str = None, 
    hf_repo: str = "hugggof/vampnetv2",
    version_tag: str = "latest",
):
    bundle = VampNetTrainer.load_from_checkpoint(ckpt) 

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


if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        export_model()