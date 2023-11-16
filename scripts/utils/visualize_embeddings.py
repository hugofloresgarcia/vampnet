"""
TODO: train a linear probe
usage:
   python gtzan_embeddings.py --args.load conf/interface.yml --Interface.device cuda --path_to_audio /path/to/audio/labels  --output_dir /path/to/output
"""
from pathlib import Path
from typing import List

import audiotools as at
from audiotools import AudioSignal
import argbind
import torch
import numpy as np
import zipfile
import json

from vampnet.interface import Interface
import tqdm

# bind the Interface to argbind
Interface = argbind.bind(Interface)

DEBUG = False


def smart_plotly_export(fig, save_path: Path):
    img_format = save_path.suffix[1:]
    if img_format == "html":
        fig.write_html(save_path)
    elif img_format == 'bytes':
        return fig.to_image(format='png')
    #TODO: come back and make this prettier
    elif img_format == 'numpy':
        import io
        from PIL import Image

        def plotly_fig2array(fig):
            #convert Plotly fig to  an array
            fig_bytes = fig.to_image(format="png", width=1200, height=700)
            buf = io.BytesIO(fig_bytes)
            img = Image.open(buf)
            return np.asarray(img)

        return plotly_fig2array(fig)
    elif img_format == 'jpeg' or 'png' or 'webp':
        fig.write_image(save_path)
    else:
        raise ValueError("invalid image format")


def dim_reduce(annotated_embeddings, layer, output_dir, n_components=3, method="tsne"):
    """
    dimensionality reduction for visualization!
    saves an html plotly figure to save_path
    parameters:
        annotated_embeddings (list): the annotated enmbeddings to be reduced; embeddings have shape (samples, features)
        labels (list): list of labels for embedding
        save_path (str): path where u wanna save ur figure
        method (str): umap, tsne, or pca
        title (str): title for ur figure
    returns:
        proj (np.ndarray): projection vector with shape (samples, dimensions)
    """
    import pandas as pd
    import plotly.express as px

    fig_name = f"vampnet-embeddings-layer={layer}"
    fig_title = f"{fig_name}_{method}"
    save_path = (output_dir / fig_name).with_suffix(".html")

    if method == "umap":
        reducer = umap.UMAP(n_components=n_components)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"invalid method: {method}")

    labels = [emb.label for emb in annotated_embeddings]
    names = [emb.filename for emb in annotated_embeddings]
    embs = [emb.embedding for emb in annotated_embeddings]
    embs_at_layer = np.stack(embs)[:, layer, :]
    projs = reducer.fit_transform(embs_at_layer)

    df = pd.DataFrame(
        {
            "label": labels,
            "name": names,
            "x": projs[:, 0],
            "y": projs[:, 1],
        }
    )
    if n_components == 2:
        fig = px.scatter(
            df, x="x", y="y", color="label", hover_name="name", title=fig_title,
        )

    elif n_components == 3:
        df['z'] = projs[:, 2]
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", color="label", hover_name="name", title=fig_title
        )
    else:
        raise ValueError(f"can't plot {n_components} components")

    fig.update_traces(
        marker=dict(size=6, line=dict(width=1, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )

    return smart_plotly_export(fig, save_path)



# per JukeMIR, we want the emebddings from the middle layer?
def vampnet_embed(sig: AudioSignal, interface: Interface, layer=10):
    with torch.inference_mode():
        # preprocess the signal
        sig = interface.preprocess(sig)

        # get the coarse vampnet model
        vampnet = interface.coarse

        # get the tokens
        z = interface.encode(sig)[:, :vampnet.n_codebooks, :]
        z_latents = vampnet.embedding.from_codes(z, interface.codec)

        # do a forward pass through the model, get the embeddings
        _z, embeddings = vampnet(z_latents, return_activations=True)
        # print(f"got embeddings with shape {embeddings.shape}")
        # [layer, batch, time, n_dims]
        # [20, 1, 600ish, 768]


        # squeeze batch dim (1 bc layer should be dim 0)
        assert embeddings.shape[1] == 1, f"expected batch dim to be 1, got {embeddings.shape[0]}"
        embeddings = embeddings.squeeze(1)

        num_layers = embeddings.shape[0]
        assert layer < num_layers, f"layer {layer} is out of bounds for model with {num_layers} layers"

        # do meanpooling over the time dimension
        embeddings = embeddings.mean(dim=-2)
        # [20, 768]

        # return the embeddings
        return embeddings

from dataclasses import dataclass, fields
@dataclass
class AnnotatedEmbedding:
    label: str
    filename: str
    embedding: np.ndarray

    def save(self, path):
        """Save the Embedding object to a given path as a zip file."""
        with zipfile.ZipFile(path, 'w') as archive:

            # Save numpy array
            with archive.open('embedding.npy', 'w') as f:
                np.save(f, self.embedding)

            # Save non-numpy data as json
            non_numpy_data = {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'embedding'}
            with archive.open('data.json', 'w') as f:
                f.write(json.dumps(non_numpy_data).encode('utf-8'))

    @classmethod
    def load(cls, path):
        """Load the Embedding object from a given zip path."""
        with zipfile.ZipFile(path, 'r') as archive:

            # Load numpy array
            with archive.open('embedding.npy') as f:
                embedding = np.load(f)

            # Load non-numpy data from json
            with archive.open('data.json') as f:
                data = json.loads(f.read().decode('utf-8'))

        return cls(embedding=embedding, **data)


@argbind.bind(without_prefix=True)
def main(
    path_to_audio: str = None,
    cache_dir: str = "./.emb_cache",
    output_dir: str = "./vampnet_embeddings",
    layers: List[int] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
    method: str = "tsne",
    n_components: int = 2,
):
    path_to_audio = Path(path_to_audio)
    assert path_to_audio.exists(), f"{path_to_audio} does not exist"

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # load our interface
    # argbind will automatically load the default config,
    interface = Interface()

    # we expect path_to_audio to consist of a folder for each label, so let's get the list of labels
    labels = [Path(x).name for x in path_to_audio.iterdir() if x.is_dir()]
    print(f"Found {len(labels)} labels")
    print(f"labels: {labels}")

    # collect audio files, labels, and embeddings
    annotated_embeddings = []
    for label in labels:
        audio_files = list(at.util.find_audio(path_to_audio / label))
        print(f"Found {len(audio_files)} audio files for label {label}")

        for audio_file in tqdm.tqdm(audio_files, desc=f"embedding label {label}"):
            # check if we have a cached embedding for this file
            cached_path = cache_dir / f"{label}_{audio_file.stem}.emb"
            if cached_path.exists():
                # if so, load it
                if DEBUG:
                    print(f"loading cached embedding for {cached_path.stem}")
                embedding = AnnotatedEmbedding.load(cached_path)
            else:
                try:
                    sig = AudioSignal(audio_file)
                except Exception as e:
                    print(f"failed to load {audio_file.name} with error {e}")
                    print(f"skipping {audio_file.name}")
                    continue

                # gets the embedding
                emb = vampnet_embed(sig, interface).cpu().numpy()

                # create an embedding we can save/load
                embedding = AnnotatedEmbedding(
                    label=label, filename=audio_file.name, embedding=emb
                )

                # cache the embeddings
                cached_path.parent.mkdir(exist_ok=True, parents=True)
                embedding.save(cached_path)
            annotated_embeddings.append(embedding)

    # now, let's do a dim reduction on the embeddings and visualize them.
    for layer in tqdm.tqdm(layers, desc="dim reduction"):
        dim_reduce(
            annotated_embeddings,
            layer,
            output_dir=output_dir,
            n_components=n_components,
            method=method,
        )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        main()
