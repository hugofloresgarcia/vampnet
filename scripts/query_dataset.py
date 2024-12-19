
import soundmaterial as sm
import pandas as pd

db_path = "scratch/data-fast/sm.db"
query = "SELECT * FROM audio_file JOIN dataset where dataset.name = 'vctk'"

conn = sm.connect(db_path)
print(f"loading data from {db_path}")
df = pd.read_sql(query, conn)

# what's the average duration of an audio file here? 
lengths = df["duration"].values
print(f"average duration: {lengths.mean()}")

# let's split durations into quartiles
quartiles = pd.qcut(lengths, 4)
print(f"quartiles: {quartiles}")

# or we can split into 10 equal parts
deciles = pd.qcut(lengths, 10)
print(f"deciles: {deciles}")

# or 100 equal parts
percentiles = pd.qcut(lengths, 100)
print(f"percentiles: {percentiles}")

# what's the 5th percentile?
print(f"5th percentile: {percentiles.categories[5]}")

