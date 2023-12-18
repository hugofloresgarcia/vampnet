from pathlib import Path
import pandas as pd
import numpy as np
from typing import List

from vampnet.condition import ConditionFeatures, YamnetConditioner
from vampnet.util import AnnotatedEmbedding, dim_reduce
from tqdm import tqdm
import argbind

@argbind.bind(without_prefix=True)
def main(
    input_csv: str = None, 
    output_dir: str = "data/artifacts/salad_bowl_viz"
):
    # load the data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # load the yamnet conditioner
    yamnet = YamnetConditioner()

    viz_embeddings = []

    # lets go through each row
    pbar = tqdm(df.iterrows(), total=len(df))
    for _, row in pbar:
        # get the yamnet path
        yamnet_path = Path(row["yamnet_root"]) / Path(row["yamnet_path"])

        # load the yamnet features
        yamnet_features = ConditionFeatures.load(yamnet_path)

        # get the scores
        scores = yamnet_features.features['scores']

        # get the embeddings
        embeddings = yamnet_features.features['embeddings']

        for embedding, score in zip(embeddings, scores):
            # get the label through argmax
            label = np.argmax(score)

            # get the class name
            class_name = yamnet.class_names[label]

            # create the annotated embedding
            annotated_embedding = AnnotatedEmbedding(
                embedding=embedding,
                label=class_name, 
                filename=str(yamnet_path),
            )
            viz_embeddings.append(annotated_embedding)
        # if _ > 1000:
        #     break

    print(f"Got {len(viz_embeddings)} annotated embeddings")
    # now we have a list of annotated embeddings
    # we can visualize them

    dim_reduce(viz_embeddings, output_dir, n_components=2, method="tsne")

        

if __name__ == "__main__":

    args = argbind.parse_args()

    with argbind.scope(args):
        main()