from typing import List
from pathlib import Path
import audiotools as at
from audiotools import util
import argbind
import tqdm
import pandas as pd
import plotly.express as px
import tensorflow_hub as hub
import tensorflow as tf
import csv
import shutil  # For copying files



def load_model():
    import io
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = model.class_map_path().numpy()
    class_map_csv_text = tf.io.read_file(class_map_path).numpy().decode('utf-8')
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return model, class_names

model, class_names = load_model()


def yamnet_tag(sig: at.AudioSignal) -> List[str]:
    sig = sig.resample(16000).to_mono()
    scores, embeddings, spectrogram = model(sig.samples[0, 0, :].cpu().numpy())
    scores.shape.assert_is_compatible_with([None, 521])

    # Get the indices of the top 3 scores
    top_5_indices = scores.numpy().mean(axis=0).argsort()[-5:][::-1]
    top_5 = [class_names[i] for i in top_5_indices]

    return top_5, embeddings

def plot_cumulative_duration(df: pd.DataFrame, output_dir: str):
    tags_df = df.explode("tags")
    cumulative_duration = tags_df.groupby('tags')['duration'].sum().sort_values(ascending=False)
    
    fig = px.bar(cumulative_duration, x=cumulative_duration.index, y=cumulative_duration.values, 
                 labels={'x':'Tag', 'y':'Cumulative Duration (seconds)'}, 
                 title='Cumulative Duration for Each Tag')
    
    fig.update_layout(xaxis_tickangle=-45)
    fig.write_image(str(output_dir / "cumulative_duration_plot.png"))
    fig.write_html(str(output_dir / "cumulative_duration_plot.html"))

def plot_sample_rate_histogram(df: pd.DataFrame, output_dir: str):
    fig = px.histogram(df, x="sample_rate", title='Sample Rate Histogram')
    fig.write_image(str(output_dir / "sample_rate_histogram.png"))
    fig.write_html(str(output_dir / "sample_rate_histogram.html"))


def plot_duration_boxplot(df: pd.DataFrame, output_dir: str):
    """
    Creates boxplots showing the distribution of song durations.
    1. Overall song duration distribution.
    2. Song duration distribution grouped by tags.
    """
    # Plot overall song duration distribution
    fig_overall = px.box(df, y="duration", title="Overall Song Duration Distribution")
    fig_overall.write_image(str(output_dir / "overall_duration_boxplot.png"))
    fig_overall.write_html(str(output_dir / "overall_duration_boxplot.html"))
    
    # Plot song duration distribution grouped by tags
    tags_df = df.explode("tags")
    fig_by_tag = px.box(tags_df, x="tags", y="duration", title="Song Duration Distribution by Tag")
    fig_by_tag.update_layout(xaxis_tickangle=-45)
    fig_by_tag.write_image(str(output_dir / "duration_by_tag_boxplot.png"))
    fig_by_tag.write_html(str(output_dir / "duration_by_tag_boxplot.html"))


def copy_representative_files(df: pd.DataFrame, output_dir: Path, max_files: int = 500):
    """
    Copies a representative sample of files based on the tags to the output directory,
    grouped by their respective tags.
    """
    tags_df = df.explode("tags")
    tag_counts = tags_df['tags'].value_counts()
    total_tags = tag_counts.sum()
    
    copied_files_count = 0

    # For each tag, calculate the proportional count of files to copy.
    for tag, count in tag_counts.items():
        num_files_for_tag = min(int((count / total_tags) * max_files), max_files - copied_files_count)
        
        sample_files = tags_df[tags_df['tags'] == tag].sample(num_files_for_tag)

        if not sample_files.empty:
            # Create a directory for the tag only if there are files for that tag
            tag_dir = output_dir / tag
            tag_dir.mkdir(parents=True, exist_ok=True)

            for _, row in sample_files.iterrows():
                destination = tag_dir / row['name']
                shutil.copy(row['filename'], destination)
        
        copied_files_count += num_files_for_tag

        if copied_files_count >= max_files:
            break

@argbind.bind(without_prefix=True)
def inspect_dataset(
    folder: str = None, 
):
    assert folder is not None, "folder must be specified"

    output_dir = Path(Path(folder).name + "-metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"finding audio...")
    audio_files = at.util.find_audio(folder)
    print(f"found {len(audio_files)} audio files")
    metadata = []
    for file in tqdm.tqdm(audio_files):
        info = at.util.info(file)
        sig = at.AudioSignal(file)
        tags, embeddings = yamnet_tag(sig)
        breakpoint()
        meta = {
            "duration": info.duration, 
            "sample_rate": info.sample_rate,
            "filename": file,
            "name": Path(file).name,
            "tags": tags, 

        }
        metadata.append(meta)

    df = pd.DataFrame(metadata)

    plot_cumulative_duration(df, output_dir)
    plot_sample_rate_histogram(df, output_dir)
    plot_duration_boxplot(df, output_dir)
    df.to_csv(output_dir / f"metadata.csv", index=False)

    # Copy the representative files
    sample_output_dir = output_dir /  "samples"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    copy_representative_files(df, sample_output_dir)

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        inspect_dataset()
