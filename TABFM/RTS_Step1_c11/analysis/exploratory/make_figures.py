#!/usr/bin/env python3
"""
Publication-ready data figures for the C11 total-hip BMD TabPFN-3 Aim-1 study + probes.
Conference/slides venue. Colorblind-safe (Okabe-Ito), Liberation Sans, PNG(300dpi)+SVG.
All numbers read from verified result CSVs; no new analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ---- style ----
plt.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'           # editable text in SVG
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.size'] = 11
# Okabe-Ito colorblind-safe
OI = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
      'vermillion': '#D55E00', 'sky': '#56B4E9', 'purple': '#CC79A7',
      'yellow': '#F0E442', 'black': '#000000', 'grey': '#999999'}

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'bmd')
FIG = f'{BASE}/figures'
os.makedirs(FIG, exist_ok=True)

def save(fig, name):
    png = f'{FIG}/{name}.png'
    svg = f'{FIG}/{name}.svg'
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)

# ---- load data ----
metrics = pd.read_csv(f'{BASE}/tfm_results_metrics.csv')
null = pd.read_csv(f'{BASE}/tfm_results_null.csv')
null_e = pd.read_csv(f'{BASE}/tfm_null_distribution_everything.csv')
null_s = pd.read_csv(f'{BASE}/tfm_null_distribution_shortlist.csv')
null_ood = pd.read_csv(f'{BASE}/tfm_null_distribution_everything_ood.csv')
ood = pd.read_csv(f'{BASE}/tfm_results_null_ood.csv')
oof = pd.read_csv(f'{BASE}/tfm_oof_predictions_everything.csv')
gi = pd.read_csv(f'{BASE}/tfm_group_importance.csv')
ws = pd.read_csv(f'{BASE}/tfm_width_sweep.csv')
wsum = pd.read_csv(f'{BASE}/tfm_width_sweep_summary.csv')
audit = pd.read_csv(f'{BASE}/c11_totalhipBMD_target_audit.csv')
fdict = pd.read_csv(f'{BASE}/c11_tfm_feature_dictionary.csv')

# ============ F2: design data figure (target dist + coverage) ============
def f2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    y = audit['totalhip_BMD_change'].values
    ax = axes[0]
    ax.hist(y, bins=14, color=OI['blue'], edgecolor='white', alpha=0.9)
    ax.axvline(0, color=OI['black'], lw=1.0, ls='--')
    ax.axvline(y.mean(), color=OI['vermillion'], lw=1.6)
    ax.set_xlabel('Total-hip BMD change (g/cm²)')
    ax.set_ylabel('Subjects')
    ax.set_title('A  Target distribution (n=38)', loc='left', fontweight='bold')
    ax.text(0.02, 0.97, f'mean {y.mean():.4f}\nsd {y.std():.4f}\n22/38 net loss',
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', fc='#f2f2f2', ec='#cccccc'))
    ax = axes[1]
    npres = fdict['n_present'].values
    ax.hist(npres, bins=np.arange(0, 40, 2), color=OI['green'], edgecolor='white', alpha=0.9)
    ax.set_xlabel('Subjects with a value (of 38)')
    ax.set_ylabel('Features')
    ax.set_title('B  Feature coverage (628 features)', loc='left', fontweight='bold')
    ax.text(0.98, 0.97, f'median {int(np.median(npres))}/38\nrange {npres.min()}–{npres.max()}\nno imputation',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', fc='#f2f2f2', ec='#cccccc'))
    fig.tight_layout()
    save(fig, 'F2_design_data')

# ============ F4: permutation-null distributions ============
def f4():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ne = null[null['table'] == 'everything'].iloc[0]
    ns = null[null['table'] == 'shortlist'].iloc[0]
    for ax, dist, rec, col, name in [
        (axes[0], null_e['null_spearman'].values, ne, OI['blue'], 'Everything (628 features)'),
        (axes[1], null_s['null_spearman'].values, ns, OI['orange'], 'Shortlist (5 features)')]:
        ax.hist(dist, bins=40, color=col, edgecolor='white', alpha=0.75)
        obs = rec['obs_spearman']
        ax.axvline(obs, color=OI['vermillion'], lw=2.2)
        ax.axvline(rec['null_spearman_mean'], color=OI['black'], lw=1.4, ls='--')
        ax.axvline(rec['null_spearman_p95'], color=OI['grey'], lw=1.1, ls=':')
        ax.set_xlabel('LOOCV Spearman (null)')
        ax.set_ylabel('Shuffles')
        ax.set_title(name, loc='left', fontweight='bold')
        p = rec['emp_p_spearman']
        ax.text(0.02, 0.97,
                f'observed {obs:.3f}\nnull mean {rec["null_spearman_mean"]:.3f}\np = {p:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', fc='#f2f2f2', ec='#cccccc'))
    axes[0].annotate('', xy=(ne['obs_spearman'], axes[0].get_ylim()[1]*0.55),
                     xytext=(ne['null_spearman_mean'], axes[0].get_ylim()[1]*0.55),
                     arrowprops=dict(arrowstyle='<->', color=OI['black'], lw=1.0))
    fig.suptitle('Permutation null (1000 shuffles) vs observed', y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, 'F4_null_distributions')

# ============ F5: OOF predicted vs actual ============
def f5():
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    yt = oof['y_true'].values
    yp = oof['oof_pred_10x5'].values
    fl = oof['floor_pred'].values
    ax.scatter(yt, yp, s=42, color=OI['blue'], alpha=0.8, edgecolor='white', label='TabPFN-3 (10×5 OOF)')
    ax.scatter(yt, fl, s=30, color=OI['grey'], alpha=0.5, marker='x', label='Mean-predictor floor')
    lim = [min(yt.min(), yp.min()) - 0.005, max(yt.max(), yp.max()) + 0.005]
    ax.plot(lim, lim, color=OI['black'], lw=1.0, ls='--')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual total-hip BMD change (g/cm²)')
    ax.set_ylabel('Predicted (out-of-fold)')
    ax.set_title('Everything table — predicted vs actual', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.text(0.98, 0.02, f'pred sd {yp.std():.4f}\nactual sd {yt.std():.4f}\n(compression)',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', fc='#f2f2f2', ec='#cccccc'))
    fig.tight_layout()
    save(fig, 'F5_oof_scatter')

# ============ F6: descriptive metrics forest-style ============
def f6():
    mets = [('r2', 'R²'), ('rmse', 'RMSE'), ('mae', 'MAE'), ('spearman', 'Spearman')]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
    tables = ['everything', 'shortlist']
    cols = {'everything': OI['blue'], 'shortlist': OI['orange']}
    for ax, (key, lab) in zip(axes, mets):
        for i, t in enumerate(tables):
            r = metrics[metrics['table'] == t].iloc[0]
            v = r[key]; lo = r[f'{key}_ci_lo']; hi = r[f'{key}_ci_hi']
            ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt='o', color=cols[t],
                        capsize=4, ms=7, lw=1.6, label=t if key == 'r2' else None)
        # floor
        fv = metrics[metrics['table'] == 'everything'].iloc[0][f'floor_{key}']
        ax.axvline(fv, color=OI['grey'], lw=1.0, ls=':')
        ax.set_yticks([0, 1]); ax.set_yticklabels(['everything', 'shortlist'])
        ax.set_title(lab, loc='left', fontweight='bold')
        if key == 'r2':
            ax.axvline(0, color=OI['black'], lw=0.8, ls='-')
    axes[0].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle('Descriptive metrics (10×5 repeated CV; dotted = mean-predictor floor)', y=1.04, fontsize=12)
    fig.tight_layout()
    save(fig, 'F6_descriptive_metrics')

# ============ F7: group-importance ranking ============
def f7():
    d = gi.sort_values('obs_drop_spearman', ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ypos = np.arange(len(d))
    colors = []
    for g in d['group']:
        if g == 'Screening_fitness':
            colors.append(OI['vermillion'])
        elif g == 'Bone_mineral_regional':
            colors.append(OI['sky'])
        else:
            colors.append(OI['blue'])
    ax.barh(ypos, d['obs_drop_spearman'], color=colors, edgecolor='white', alpha=0.9)
    # null mean ± sd overlay
    ax.errorbar(d['null_drop_spearman_mean'], ypos,
                xerr=d['null_drop_spearman_sd'], fmt='none',
                ecolor=OI['black'], elinewidth=0.8, capsize=2, alpha=0.6)
    ax.axvline(0, color=OI['black'], lw=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{g} ({n})' for g, n in zip(d['group'], d['n_features'])], fontsize=8.5)
    ax.set_xlabel('Drop in LOOCV Spearman when group permuted\n(bars = observed; whiskers = shuffle-null mean ± sd)')
    ax.set_title('Grouped permutation importance (16 domain groups)', loc='left', fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=OI['vermillion'], label='Screening_fitness (driver)'),
                       Patch(fc=OI['sky'], label='Bone_mineral_regional (null)'),
                       Patch(fc=OI['blue'], label='other groups')],
              frameon=False, fontsize=8.5, loc='lower right')
    fig.tight_layout()
    save(fig, 'F7_group_importance')

# ============ F8: width-sweep curve ============
def f8():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sub = ws[ws['width'] < 628]
    # individual draws
    for w, grp in sub.groupby('width'):
        ax.scatter(np.full(len(grp), w) + np.random.default_rng(1).normal(0, 6, len(grp)),
                   grp['spearman'], s=34, color=OI['sky'], alpha=0.8, edgecolor='white', zorder=3)
    # mean line
    m = wsum[wsum['width'] < 628]
    ax.errorbar(m['width'], m['spearman_mean'], yerr=m['spearman_sd'], fmt='-o',
                color=OI['blue'], lw=1.8, ms=7, capsize=4, label='subsample mean ± sd', zorder=4)
    # full table
    full = wsum[wsum['width'] == 628].iloc[0]
    ax.scatter([628], [full['spearman_mean']], s=130, color=OI['vermillion'], marker='*',
               edgecolor='black', zorder=5, label='full table (628)')
    ax.axhline(-0.393, color=OI['grey'], lw=1.1, ls='--')
    ax.text(8, -0.37, 'everything null center (−0.39)', fontsize=8, color=OI['grey'], va='bottom')
    ax.axhline(0, color=OI['black'], lw=0.7, ls=':')
    ax.set_xscale('symlog', linthresh=50)
    ax.set_xticks([5, 25, 100, 250, 628]); ax.set_xticklabels(['5', '25', '100', '250', '628'])
    ax.set_xlabel('Number of features (coverage-matched subsample)')
    ax.set_ylabel('LOOCV Spearman')
    ax.set_title('Feature-width sweep — signal emerges only at full width', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout()
    save(fig, 'F8_width_sweep')

# ============ F9: OOD robustness ============
def f9():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    ne = null[null['table'] == 'everything'].iloc[0]
    for ax, dist, obs, mean, p, col, name in [
        (axes[0], null_e['null_spearman'].values, ne['obs_spearman'], ne['null_spearman_mean'],
         ne['emp_p_spearman'], OI['blue'], 'Default checkpoint'),
        (axes[1], null_ood['null_spearman'].values, ood.iloc[0]['obs_spearman'],
         ood.iloc[0]['null_spearman_mean'], ood.iloc[0]['emp_p_spearman'], OI['green'], 'OOD checkpoint')]:
        ax.hist(dist, bins=40 if len(dist) > 300 else 25, color=col, edgecolor='white', alpha=0.75)
        ax.axvline(obs, color=OI['vermillion'], lw=2.2)
        ax.axvline(mean, color=OI['black'], lw=1.4, ls='--')
        ax.set_xlabel('LOOCV Spearman (null)')
        ax.set_title(name, loc='left', fontweight='bold')
        ax.text(0.02, 0.97, f'observed {obs:.3f}\nnull mean {mean:.3f}\np = {p:.3f}',
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', fc='#f2f2f2', ec='#cccccc'))
    axes[0].set_ylabel('Shuffles')
    fig.suptitle('Robustness to TabPFN-3 checkpoint (everything table)', y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, 'F9_ood_robustness')

if __name__ == '__main__':
    f2(); f4(); f5(); f6(); f7(); f8(); f9()
    print('ALL DATA FIGURES DONE')
