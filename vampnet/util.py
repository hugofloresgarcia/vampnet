import torch
import tqdm
from einops import rearrange
from pathlib import Path
import numpy as np


def scalar_to_batch_tensor(x, batch_size):
    return torch.tensor(x).repeat(batch_size)


def parallelize(fn, *iterables, parallel: str = "thread_map", **kwargs):
    if parallel == "thread_map":
        from tqdm.contrib.concurrent import thread_map

        return thread_map(fn, *iterables, **kwargs)
    elif parallel == "process_map":
        from tqdm.contrib.concurrent import process_map

        return process_map(fn, *iterables, **kwargs)
    elif parallel == "single":
        return [fn(x) for x in tqdm.tqdm(*iterables)]
    else:
        raise ValueError(
            f"parallel must be one of 'thread_map', 'process_map', 'single', but got {parallel}"
        )


def codebook_flatten(tokens: torch.Tensor):
    """
    flatten a sequence of tokens from (batch, codebook, time) to (batch, codebook * time)
    """
    return rearrange(tokens, "b c t -> b (t c)")


def codebook_unflatten(flat_tokens: torch.Tensor, n_c: int = None):
    """
    unflatten a sequence of tokens from (batch, codebook * time) to (batch, codebook, time)
    """
    tokens = rearrange(flat_tokens, "b (t c) -> b c t", c=n_c)
    return tokens


def smart_plotly_export(fig, save_path: Path):
    img_format = save_path.suffix[1:]
    if img_format == "html":
        fig.write_html(str(save_path))
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


def dim_reduce(annotated_embeddings, output_dir, n_components=3, method="tsne", max_samples = 50000):
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

    fig_name = f"embeddings"
    fig_title = f"{fig_name}_{method}"
    save_path = (Path(output_dir) / fig_name).with_suffix(".html")
    save_path.parent.mkdir(exist_ok=True, parents=True)

    # if we have more than 50k annotated embeddings, we'll sample 50k
    
    if len(annotated_embeddings) > max_samples:
        print(f"reducing {len(annotated_embeddings)} embeddings to 50k")
        annotated_embeddings = np.random.choice(
            annotated_embeddings, size=max_samples, replace=False
        )
    if method == "umap":
        from umap import UMAP
        reducer = umap.UMAP(n_components=n_components)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, verbose=4)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"invalid method: {method}")

    labels = [emb.label for emb in annotated_embeddings]
    names = [emb.filename for emb in annotated_embeddings]
    embs = [emb.embedding for emb in annotated_embeddings]
    projs = reducer.fit_transform(np.stack(embs))

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
        marker=dict(size=12, line=dict(width=1, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    smart_plotly_export(fig, save_path)
    return smart_plotly_export(fig, save_path.with_suffix(".png"))

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
