import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False

with open('/home/cyan/default/crypto/data/results_v5.json') as f:
    v5 = json.load(f)
with open('/home/cyan/default/crypto/data/results_v4.json') as f:
    v4 = json.load(f)
with open('/home/cyan/default/crypto/data/backtest_v5_results.json') as f:
    bt5 = json.load(f)
with open('/home/cyan/default/crypto/data/backtest_v4_results.json') as f:
    bt4 = json.load(f)

OUT = '/home/cyan/default/crypto/docs/figures'

seeds = ['seed_42', 'seed_123', 'seed_456']
colors = {'seed_42': '#2563eb', 'seed_123': '#dc2626', 'seed_456': '#16a34a'}
seed_labels = {'seed_42': 'Seed 42', 'seed_123': 'Seed 123', 'seed_456': 'Seed 456'}

fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True, gridspec_kw={'hspace': 0.15})

ax = axes[0]
for s in seeds:
    h = v5['histories'][s]
    epochs = range(1, len(h['train_loss']) + 1)
    ax.plot(epochs, h['train_loss'], color=colors[s], linestyle='-', linewidth=1.2,
            label=f'{seed_labels[s]} Train Loss')
    ax.plot(epochs, h['val_loss'], color=colors[s], linestyle='--', linewidth=1.2,
            label=f'{seed_labels[s]} Val Loss')
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('V5 Training Curves (43 features + external data)', fontsize=14, fontweight='bold', pad=10)
ax.legend(fontsize=8, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

ax = axes[1]
for s in seeds:
    h = v5['histories'][s]
    epochs = range(1, len(h['val_ic']) + 1)
    ax.plot(epochs, h['val_ic'], color=colors[s], linewidth=1.2, label=seed_labels[s])
ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
ax.set_ylabel('Validation IC', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

ax = axes[2]
for s in seeds:
    h = v5['histories'][s]
    epochs = range(1, len(h['val_dir_acc']) + 1)
    ax.plot(epochs, h['val_dir_acc'], color=colors[s], linewidth=1.2, label=seed_labels[s])
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, label='Random baseline (0.5)')
ax.set_ylabel('Direction Accuracy', fontsize=12)
ax.set_xlabel('Epoch', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.47, 0.57)

plt.tight_layout()
path1 = f'{OUT}/v5_training_curves.png'
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Chart 1 saved: {path1}')

pairs = v4['config']['pairs']
pair_short = [p.replace('/USDT', '') for p in pairs]

v4_ics = [v4['per_pair'][p]['ic'] for p in pairs]
v5_ics = [v5['per_pair'][p]['ic'] for p in pairs]

x = np.arange(len(pairs))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))
bars1 = ax.bar(x - width/2, v4_ics, width, label='V4 IC', color='#2563eb', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, v5_ics, width, label='V5 IC', color='#ea580c', alpha=0.85, edgecolor='white')

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pair_short, fontsize=10, rotation=30, ha='right')
ax.set_ylabel('Information Coefficient (IC)', fontsize=12)
ax.set_title('V4 vs V5 Per-Pair Information Coefficient (IC)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

for bar in bars1:
    h = bar.get_height()
    yoff = 0.005 if h >= 0 else -0.015
    ax.text(bar.get_x() + bar.get_width()/2, h + yoff, f'{h:.2f}',
            ha='center', va='bottom' if h >= 0 else 'top', fontsize=7, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    yoff = 0.005 if h >= 0 else -0.015
    ax.text(bar.get_x() + bar.get_width()/2, h + yoff, f'{h:.2f}',
            ha='center', va='bottom' if h >= 0 else 'top', fontsize=7, fontweight='bold')

plt.tight_layout()
path2 = f'{OUT}/v4_vs_v5_ic_comparison.png'
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Chart 2 saved: {path2}')

strategies = ['LS(th=0.005)', 'LS(th=0.010)', 'Top3-LS']
strat_labels = ['LS(th=0.005)', 'LS(th=0.010)', 'Top3-LS', 'Buy&Hold']
v4_returns = [bt4['strategies'][s]['total_return_pct'] for s in strategies] + [bt4['buy_and_hold_return_pct']]
v5_returns = [bt5['strategies'][s]['total_return_pct'] for s in strategies] + [bt5['buy_and_hold_return_pct']]

x = np.arange(len(strat_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width/2, v4_returns, width, label='V4', color='#2563eb', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width/2, v5_returns, width, label='V5', color='#ea580c', alpha=0.85, edgecolor='white')

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(strat_labels, fontsize=11)
ax.set_ylabel('Total Return (%)', fontsize=12)
ax.set_title('V4 vs V5 Strategy Returns Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

for bar in bars1:
    h = bar.get_height()
    yoff = 1.5 if h >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2, h + yoff, f'{h:.1f}%',
            ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    yoff = 1.5 if h >= 0 else -1.5
    ax.text(bar.get_x() + bar.get_width()/2, h + yoff, f'{h:.1f}%',
            ha='center', va='bottom' if h >= 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
path3 = f'{OUT}/v4_vs_v5_backtest_comparison.png'
plt.savefig(path3, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] Chart 3 saved: {path3}')

print('\nAll 3 charts generated successfully!')
