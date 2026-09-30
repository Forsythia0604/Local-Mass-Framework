# Tier C — ResNet-50 / ImageNet channel gates (phase 1)

Source of the Experiment 5 ImageNet column. Retrieved from Gadi 2026-08-24; all
files md5-verified against `/scratch/nu66/sm5828/bayes/e1/results/tierC/`
(originals left in place as a second copy).

## What produced these

Unmodified `pbs/run_e1_imagenet_phase1.pbs`, submitted as:

    qsub -N e1_<short>_s<seed> -l walltime=10:00:00 \
         -v QFAMILY=<family>,SEED=<seed> pbs/run_e1_imagenet_phase1.pbs

which runs:

    python3 -u run_e1_structured.py \
        --arch resnet50 --dataset imagenet --phase 1 --pretrained \
        --qfamily <family> --slab gaussian --pi 0.2 --kl_weight 1e-4 \
        --epochs 8 --warmup_epochs 0 --anneal_epochs 4 --gate_lr 0.05 \
        --batch_size 192 --seed <seed> --device cuda --data_dir $PBS_JOBFS/imagenet

Identical settings for all four families — only `--qfamily` and `--seed` differ, so
the comparison isolates the variational family. 7552 gate channels; prior atom mass
`1 - pi = 0.8`. Base network frozen (gates only).

**Phase 2 was deliberately NOT run.** It is KD fine-tuning, which E1 excludes as a
separate project. Exp 5 needs only the phase-1 gate checkpoint.

## Runs

| family | seeds | job IDs | test acc | atom | atom mass |
|---|---|---|---|---|---|
| gaussian     | 1,2,3 | (s1 Jun-26), 177191675-6 | 75.20-75.51 | no  | 0 |
| horseshoe    | 1,2,3 | 177191669-71             | 73.85-74.49 | no  | 0 |
| spikeslab    | 1,2,3 | (s1 Jun-26), 177191677-8 | 71.42-72.13 | yes | 0.0632-0.0659 |
| hardconcrete | 1,2,3 | 177191672-4              | 67.59-68.90 | yes | 0.0231-0.0238 |

Seeds 1 for gaussian/spikeslab date from 2026-06-26; all others 2026-08-24.
All 10 new jobs exited 0. Cost ~160 SU / ~4h35m each (see `../../pbs_logs/*.o*`).

## Files

- `r50_<family>_p1_s<seed>.json` — hyperparameters, accuracy, mass-index summary
- `r50_<family>_p1_s<seed>.log`  — stdout
- `../../their_style/imagenet_ckpts/r50_<family>_p1_s<seed>_gnet.pt` — gate checkpoint

## To rebuild the Exp 5 ImageNet figures

Marginal extraction runs locally on CPU (no ImageNet data, no GPU):

    cd ../../their_style
    python structured_marginals.py --arch resnet50 --families gaussian horseshoe \
        spikeslab hardconcrete --seed <seed> --device cpu \
        --out marginals_imagenet_s<seed>.npz
    python local_mass_nn.py --mode trained --npz marginals_imagenet_s<seed>.npz \
        --pi 0.2 --slab_loc 1.0 --grid_max 1.2 --tag _imagenet

SAVE THE .npz INTO THE REPO — the previous round's marginals were never kept, which
forced this whole re-run.
