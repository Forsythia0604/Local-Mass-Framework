"""Reflow existing vector curves without fitting or inventing experiment data.

The neural-network plots retain the supplied PDF polyline coordinates exactly
after an affine change of plot-area coordinates. Tick labels and legends are
typeset afresh for a half-column panel. These are NOT recovered raw results.
Figure 3 is separately recomputed from the exact RE-KL formulas in Appendix C.3.
Dependencies: PyMuPDF, numpy, matplotlib. Run from any working directory.
"""
from pathlib import Path
import json
import fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results' / 'figure_sources'
OUT = ROOT / 'results' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.size': 6.5, 'axes.labelsize': 6.5, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 5.8, 'axes.linewidth': .55,
    'lines.linewidth': .9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'xtick.major.width': .5, 'ytick.major.width': .5,
})

def compact(name):
    doc = fitz.open(SOURCE / (name + '.pdf'))
    page = doc[0]
    drawings = page.get_drawings()
    # The second white rectangle in each supplied Matplotlib PDF is the axes.
    rect = drawings[1]['rect']
    curves = [p for p in drawings if p['type'] == 's'
              and len(p['items']) > 10 and p['rect'].width > .9 * rect.width
              and p.get('color') is not None and p['color'] != (0., 0., 0.)]
    mass = 'small_ball' in name
    expected = 5 if mass else 3
    assert len(curves) == expected, (name, len(curves))
    fig = plt.figure(figsize=(1.70, 1.95))
    ax = fig.add_axes([.24, .40, .72, .57])
    labels = (['Prior', 'Gaussian', 'Student-$t$', 'Spike-slab', 'Hard-concr.']
              if mass else [r'$q\|p$ (G)', r'$p\|q$ (G)', r'$p\|q$ (SS)'])
    handles = []
    for i, path in enumerate(curves):
        pts = [path['items'][0][1]]
        for item in path['items']:
            assert item[0] == 'l', (name, item[0])
            pts.append(item[2])
        x = np.array([(p.x - rect.x0) / rect.width for p in pts])
        y = np.array([(rect.y1 - p.y) / rect.height for p in pts])
        styles = ['-.', '-', '--', '-', '--'] if mass else ['-', '--', ':']
        h, = ax.plot(x, y, color=path['color'], ls=styles[i], label=labels[i])
        handles.append(h)
    # Match each outside logarithmic tick label to its original tick mark.
    xticks, yticks = [], []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            spans = line['spans']
            if line['dir'] != (1., 0.) or len(spans) != 2 or spans[0]['text'] != '10':
                continue
            exp = spans[1]['text'].replace('\u2212', '-')
            if not exp.lstrip('-').isdigit():
                continue
            bbox = fitz.Rect(line['bbox'])
            if bbox.y0 > rect.y1:
                centre = (bbox.x0 + bbox.x1) / 2
                ticks = [p['rect'].x0 for p in drawings if p['type'] == 'fs'
                         and p['rect'].width < .01 and abs(p['rect'].y0 - rect.y1) < .05]
                value = min(ticks, key=lambda t: abs(t - centre))
                xticks.append(((value - rect.x0) / rect.width, exp))
            elif bbox.x1 < rect.x0:
                centre = (bbox.y0 + bbox.y1) / 2
                ticks = [p['rect'].y0 for p in drawings if p['type'] == 'fs'
                         and p['rect'].height < .01 and abs(p['rect'].x1 - rect.x0) < .05]
                value = min(ticks, key=lambda t: abs(t - centre))
                yticks.append(((rect.y1 - value) / rect.height, exp))
    # Retain every x decade and at most four labelled y decades.
    if len(yticks) > 4:
        yticks = [yticks[i] for i in np.unique(np.linspace(0, len(yticks)-1, 4).round().astype(int))]
    assert len(xticks) >= 3 and len(yticks) >= 3
    ax.set_xticks([x[0] for x in xticks], [rf'$10^{{{x[1]}}}$' for x in xticks])
    ax.set_yticks([x[0] for x in yticks], [rf'$10^{{{x[1]}}}$' for x in yticks])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=r'Radius $r$')
    ax.set_ylabel(r'$q(B_r(0))$' if mass else r'Normalised RE-KL', labelpad=1)
    ax.tick_params(length=2, pad=1)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.51, .005),
               ncol=2 if mass else 1, frameon=False, handlelength=1.5,
               columnspacing=.55, handletextpad=.35, labelspacing=.3)
    fig.savefig(OUT / (name + '.pdf'))
    plt.close(fig)
    doc.close()
    return {'figure': name, 'curves_preserved': len(curves),
            'transformation': 'affine reflow of supplied vector path coordinates'}

reports = [compact(name) for name in [
    'fig_nn_small_ball', 'fig_nn_directional_rekl',
    'fig_nn_small_ball_cifar', 'fig_nn_small_ball_imagenet',
    'fig_nn_directional_rekl_cifar', 'fig_nn_directional_rekl_imagenet']]

alpha = .75
r = np.geomspace(1.e-4, 1., 400)
qp = 1 + alpha/(1-alpha)*r**(-.5) - 2**(-alpha)/((1-alpha)*(1-alpha/2))*r**(-alpha/2)
pq = 1 + alpha/(1-alpha)*r**.5 - 2**alpha/((1-alpha)*(1+alpha))*r**(alpha/2)
assert np.all(qp > 0) and np.all(pq > 0)
fig = plt.figure(figsize=(1.70, 1.65))
ax = fig.add_axes([.24, .29, .72, .67])
ax.loglog(r, pq, color='#1f77b4', label=r'$p\|q$')
ax.loglog(r, qp, '--', color='#d95f02', label=r'$q\|p$')
ax.set(xlabel=r'Radius $r$', ylabel='Normalised RE-KL', xlim=(1e-4,1))
ax.set_xticks([1e-4,1e-2,1])
ax.set_yticks([.1,1,10,100])
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_locator(NullLocator())
ax.tick_params(length=2, pad=1)
ax.yaxis.labelpad = 1
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.55, 0), ncol=2,
           frameon=False, handlelength=1.4, columnspacing=.7, handletextpad=.4)
fig.savefig(OUT / 'figure3_local_rekl_directionality.pdf')
plt.close(fig)
(OUT/'curve_provenance.json').write_text(json.dumps(reports, indent=2)+'\n')
print('Prepared 7 half-column vector figures. Original NN curve geometry retained.')
