

python s/00_db_init.py

python s/01_create_audio_dataset.py --audio_folder "data/prosound/Anns Animals" --dataset_name anns-animals

python s/02_preprocess.py --dataset anns-animals 

python s/04_partition.py --dataset anns-animals --train 0.9 --val 0.05 --test 0.05

# python s/03_caption.py --dataset anns-animals

