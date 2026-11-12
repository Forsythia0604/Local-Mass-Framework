# Optional ImageNet checkpoints

ImageNet gate checkpoints are not included. The saved-marginal analysis uses
`data/marginals/` and does not require checkpoints.

For checkpoint extraction, place compatible gate checkpoints here or pass a
different directory using `--ckpt_dir`.

Expected filenames: `r50_<family>_p1_s<seed>_gnet.pt`, with families `gaussian`,
`horseshoe`, `spikeslab`, `hardconcrete` and seeds `1`, `2`, `3`.

See the [neural-network usage guide](../../experiments/neural_networks/README.md)
for training and extraction commands. Checkpoint files placed here are ignored
by Git.
