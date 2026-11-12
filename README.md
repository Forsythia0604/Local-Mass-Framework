# Beyond Global Divergences
### A Local-Mass Perspective on Bayesian Inference

**Hanli Xu** (UNSW Sydney) · **Fengxiang He** (University of Edinburgh) · **Sarat Moka** (UNSW Sydney)

[Read the paper](main.pdf) · [Browse the figures](results/figures/) · [Run the experiments](code/README.md) · [View the LaTeX source](main.tex)

Global measures such as KL divergence can miss how probability mass behaves near a particular point. This paper uses Mass Indices and local regularised extended KL (RE-KL) to study when Bayesian updating and posterior approximation preserve that local behaviour. The experiments range from controlled examples and UCI logistic regression to neural-network marginals on MNIST, CIFAR-10, and ImageNet.

## Run the experiments

For the controlled and UCI experiments, use Python 3.11 and run these commands from the repository root:

```sh
cd code
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-verified.txt
python experiments/local_mass_experiments.py
```

To analyse the three saved ImageNet marginal inputs, run the following from `code/` with the same environment. This analysis does not require the ImageNet dataset, model checkpoints, or a GPU.

```sh
python experiments/neural_networks/experiment_scripts/imagenet_multiseed.py \
  --seeds 1 2 3 --marg_dir data/marginals
```

The [experiment guide](code/README.md) gives output paths and options. The [neural-network guide](code/experiments/neural_networks/README.md) explains retraining and marginal extraction. The original MNIST and CIFAR-10 marginals and the ImageNet gate checkpoints are not included, so fresh training is separate from rerunning the saved-input analysis.

## Repository layout

| Path | What it contains |
| --- | --- |
| [`main.pdf`](main.pdf) | Paper and appendices, with links to sections, figures, and references |
| [`main.tex`](main.tex), [`references.bib`](references.bib) | Complete LaTeX manuscript and bibliography |
| [`code/`](code/) | Experiment scripts, dependencies, and saved ImageNet marginals |
| [`results/`](results/) | Paper figures, original vector panels, and UCI result summaries |
| [`scripts/`](scripts/) | Paper build and individual figure regeneration |

## Build the paper

With `latexmk`, pdfLaTeX, and BibTeX installed, run:

```sh
sh scripts/build.sh
```

The build uses the included bibliography and figures and writes `main.pdf` in the repository root. It does not run the experiments. The four composite or aggregate figures used in the paper are included as vector PDFs. To regenerate the retained individual panels from the supplied vector inputs, run `python3 scripts/compact_figures.py` with PyMuPDF, NumPy, and Matplotlib installed; this does not rebuild the composites or rerun training.

## Cite this work

Hanli Xu, Fengxiang He, and Sarat Moka. *Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference.*

See [third-party and checkpoint notes](code/THIRD_PARTY.md) for asset provenance and licensing.
