alias trun="python -m torch.distributed.run --nproc_per_node gpu"
alias tdebug="CUDA_VISIBLE_DEVICES=0 python -m pdb -c c"
# alias debug=
alias dbsummary="./duckdb data/spotdl.db -readonly"