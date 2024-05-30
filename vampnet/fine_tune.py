import vampnet
import yapecs

import audiotools as at

from vampnet.train import train


def fine_tune(dataset: str, model_name: str,  **kwargs):
    with at.ml.Accelerator(amp=vampnet.AMP) as accel:
        if accel.local_rank != 0:
            sys.tracebacklimit = 0
        return train(accel, 
            save_path=vampnet.RUNS_DIR / f"finetuned-{dataset}",
            dataset=dataset, 
            model_name=model_name,
            fine_tune=True, **kwargs)



if __name__ == "__main__":
    parser = yapecs.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="dataset to fine-tune on")
    parser.add_argument("--model_name", type=str, default="vampnet-base-best", help="model to fine-tune")
    parser.add_argument("--num_iters", type=int, default=20000, help="number of steps to fine-tune")

    args = parser.parse_args()
    assert args.dataset is not None, "Please provide a dataset to fine-tune on"
    
    fine_tune(**vars(args), cli=True)


