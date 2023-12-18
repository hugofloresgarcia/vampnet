import argbind
import audiotools as at
from audiotools import util
from pathlib import Path
import pandas as pd
import tqdm
from tqdm.contrib.concurrent import thread_map
import plotly.express as px
import shutil

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
    fig_overall = px.violin(df, y="duration", box=True, points='outliers', title="Overall Song Duration Distribution")
    
    # Adjusting box whisker distance for outlier consideration
    fig_overall.update_traces(box_visible=True, line_color='black',)
    
    fig_overall.write_image(str(output_dir / "overall_duration_violin.png"))
    fig_overall.write_html(str(output_dir / "overall_duration_violin.html"))

    # write a file with the TOTAL duration in hours
    total_duration = df['duration'].sum()
    with open(output_dir / "total_duration.txt", 'w') as f:
        f.write(f"{total_duration / 3600:.2f} hours")
    
    # check if we have tags in the df 
    if 'tags' in df.columns:
        # Plot song duration distribution grouped by tags
        tags_df = df.explode("tags")
        fig_by_tag = px.box(tags_df, x="tags", y="duration", title="Song Duration Distribution by Tag")
        fig_by_tag.update_layout(xaxis_tickangle=-45)
        fig_by_tag.write_image(str(output_dir / "duration_by_tag_boxplot.png"))
        fig_by_tag.write_html(str(output_dir / "duration_by_tag_boxplot.html"))


def plot_num_channels_histogram(df: pd.DataFrame, output_dir: str):
    """
    Creates a histogram showing the distribution of the number of channels.
    """
    fig = px.histogram(df, x="num_channels", title='Number of Channels Histogram')
    fig.write_image(str(output_dir / "num_channels_histogram.png"))
    fig.write_html(str(output_dir / "num_channels_histogram.html"))

def copy_representative_files(df: pd.DataFrame, output_dir: Path, max_files: int = 100):
    """
    Copies a representative sample of files based on the tags to the output directory,
    grouped by their respective tags.
    """
    # If there are no tags, just copy the first max_files files 
    # (shuffled)
    sample_files = df.sample(max_files)

    for _, row in sample_files.iterrows():
        destination = output_dir / row['name']
        shutil.copy(row['filename'], destination)

@argbind.bind(without_prefix=True)
def extract_audio_metadata(
    input_csv: str = None,
    output_dir: str = None,
    sample_files: int = 100,
    max_workers: int = 16,
):
    # make a backup of the metadata
    shutil.copy(input_csv, f"{input_csv}.backup")

    df = pd.read_csv(input_csv)
    print(f"Loaded metadata with {len(df)} rows")

    def get_info(file, audio_too):
        try:
            info = at.util.info(file)
        except:
            print(f"Error reading {file}")
            return None

        meta = {
            "duration": info.duration, 
            "sample_rate": info.sample_rate,
            "audio_path": file,
            "name": Path(file).name,
            "num_channels": info.num_channels,
        }
        return meta


    # metadata = []
    # for file, audio_root in tqdm.tqdm(
    #         zip(df['audio_path'].to_list(), df['audio_root'].to_list()),
    #     ):

    files = [Path(r['audio_root'])/r['audio_path'] for _, r in df.iterrows()]
    metadata = thread_map(get_info, files, max_workers=max_workers)

    # remove None values
    metadata = [m for m in metadata if m is not None]

        # metadata.append(meta)

    # add the metadata to the dataframe
    metadata = pd.DataFrame(metadata)
    df = pd.concat([df, metadata], axis=1)

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    plot_sample_rate_histogram(df, artifacts_dir)
    plot_duration_boxplot(df, artifacts_dir)
    plot_num_channels_histogram(df, artifacts_dir)
    copy_representative_files(df, artifacts_dir, max_files=sample_files)
    df.to_csv(output_dir / f"metadata.csv", index=False)

    # save the total duration in hours
    total_duration = df['duration'].sum()
    with open(output_dir / "total_duration.txt", 'w') as f:
        f.write(f"{total_duration / 3600:.2f} hours")

    # overwrite the input csv with the new one, let the user know we have a backup
    print(f"done! writing to {input_csv}")
    df.to_csv(input_csv, index=False)

if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        extract_audio_metadata()

