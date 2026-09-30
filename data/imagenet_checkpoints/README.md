# Optional ImageNet checkpoints

The twelve original gate checkpoints (about 1.1 GiB in total) were intentionally
left out of this lightweight consolidation. Plotting the included ImageNet
marginals does not require them.

Expected filenames follow `r50_<family>_p1_s<seed>_gnet.pt`, for families
`gaussian`, `horseshoe`, `spikeslab`, `hardconcrete` and seeds `1`, `2`, `3`.

The source files remain in the original delivery's
`Experiments4and5/data/imagenet_checkpoints/`, beside this `Code for Git` folder's
source trees. No public download URL has been created. A future release can
provide a separate asset and checksums here.

See the neural-network guide for extraction into `data/generated/`. Checkpoint
files placed here are ignored by Git; this README remains tracked.
