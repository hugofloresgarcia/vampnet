from pathlib import Path
import json

import pandas as pd
from ast import literal_eval

import argbind


AVOID_CLASSES = ["Silence", "Mains hum"]

def contains_avoid_tags(tags_list):
    return any(tag in AVOID_CLASSES for tag in tags_list)


def build_ontology_tree(ontology):
    tree = {}
    # Step 1: Initialize nodes in the tree
    for item in ontology:
        tree[item['id']] = {
            'parent': None,
            'children': set(item['child_ids']),
            'name': item['name']
        }
    # Step 2: Set parents based on child_ids
    for item in ontology:
        for child_id in item['child_ids']:
            tree[child_id]['parent'] = item['id']
    return tree

def get_top_level_tag(tree, id_):
    while tree[id_]['parent']:
        id_ = tree[id_]['parent']
    return tree[id_]['name']

def get_majority_top_level_tag(tree, tags_ids):
    top_tags = [get_top_level_tag(tree, id_) for id_ in tags_ids]
    return max(set(top_tags), key=top_tags.count)

@argbind.bind(without_prefix=True)
def main(
    path_to_metadata: str = None, 
    ontology_path: str = "./scripts/utils/ontology.json",
    output_path: str = "./data/metadata/metadata-with-family.csv"
):
    # audioset ontology
    print(f"loading ontology from {ontology_path}")
    with open (ontology_path, "r") as f:
        ontology = json.load(f)

    print(f"loading metadata from {path_to_metadata}")
    metadata = pd.read_csv(path_to_metadata)
    print(f"found {len(metadata)} rows in metadata")
    # metadata columns: duration,sample_rate,filename,name,num_channels,tags

    print(f"building ontology tree")
    tree = build_ontology_tree(ontology)
    tag2id = {item['name']: item['id'] for item in ontology}
    id2tag = {item['id']: item['name'] for item in ontology}

    def tags2ids(_tags):
        return [tag2id[tag] for tag in _tags]

    # remove rows with AVOID_CLASSES as the top
    print(f"removing rows with tags {AVOID_CLASSES}")
    avoid_indices = metadata[metadata['tags'].apply(lambda tags: contains_avoid_tags(literal_eval(tags)))].index
    metadata.drop(avoid_indices, inplace=True)
    print(f"after removing rows with tags {AVOID_CLASSES}, there are {len(metadata)} rows left")

    # add a row to the metadata that called "family" that is the top tag of the top level of the ontology for each row in the metadata
    metadata['family'] = metadata['tags'].apply(lambda tags: get_majority_top_level_tag(tree, tags2ids(literal_eval(tags))) if tags else None)
    
    # what are the unique families and what are their file counts? 
    print(f"unique families: {metadata['family'].unique()}")
    print(f"family counts: {metadata['family'].value_counts()}")

    def get_split(filename):
        split =  filename.split("/")[2]
        assert split in ("train", "val", "test")
        return split

    metadata['split'] = metadata['filename'].apply(lambda x: get_split(x))

    # save the metadata
    print(f"saving metadata to {output_path}")
    metadata.to_csv(output_path, index=False)

    



if __name__ == "__main__":
    args = argbind.parse_args()

    with argbind.scope(args):
        main()