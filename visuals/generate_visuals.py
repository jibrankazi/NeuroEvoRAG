#!/usr/bin/env python3
"""Generate visualizations for LinkedIn post."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('/home/user/NeuroEvoRAG/visuals', exist_ok=True)

# Color palette
COLORS = {
    'baseline': '#94a3b8',
    'grid': '#f59e0b',
    'optuna': '#8b5cf6',
    'evolution': '#3b82f6',
    'random': '#10b981',
    'bg': '#0f172a',
    'text': '#e2e8f0',
    'accent': '#38bdf8',
    'grid_line': '#1e293b',
}

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['bg'],
    'axes.edgecolor': COLORS['grid_line'],
    'axes.labelcolor': COLORS['text'],
    'text.color': COLORS['text'],
    'xtick.color': COLORS['text'],
    'ytick.color': COLORS['text'],
    'grid.color': COLORS['grid_line'],
    'font.family': 'sans-serif',
    'font.size': 13,
})


# ── Visual 1: Comparison Bar Chart ──
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Hand-tuned\nBaseline', 'Grid\nSearch', 'Optuna\n(TPE)', 'Evolution', 'Random\nSearch']
scores = [0.125, 0.401, 0.431, 0.500, 0.595]
colors = [COLORS['baseline'], COLORS['grid'], COLORS['optuna'], COLORS['evolution'], COLORS['random']]

bars = ax.bar(methods, scores, color=colors, width=0.6, edgecolor='none', zorder=3)

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{score:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold',
            color=COLORS['text'])

ax.set_ylabel('Best Fitness Score', fontsize=14, fontweight='bold')
ax.set_title('RAG Hyperparameter Optimization: 4-Method Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, 0.72)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.axhline(y=0.125, color=COLORS['baseline'], linestyle='--', alpha=0.5, linewidth=1)
ax.text(4.4, 0.135, 'baseline', fontsize=10, color=COLORS['baseline'], alpha=0.7)

plt.tight_layout()
plt.savefig('/home/user/NeuroEvoRAG/visuals/1_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 1_comparison.png")


# ── Visual 2: Improvement Over Baseline ──
fig, ax = plt.subplots(figsize=(10, 5))

methods2 = ['Grid Search', 'Optuna (TPE)', 'Evolution', 'Random Search']
improvements = [221, 245, 300, 376]
colors2 = [COLORS['grid'], COLORS['optuna'], COLORS['evolution'], COLORS['random']]

bars = ax.barh(methods2, improvements, color=colors2, height=0.5, edgecolor='none', zorder=3)

for bar, imp in zip(bars, improvements):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'+{imp}%', ha='left', va='center', fontsize=14, fontweight='bold',
            color=COLORS['text'])

ax.set_xlabel('Improvement Over Hand-Tuned Baseline (%)', fontsize=13, fontweight='bold')
ax.set_title('All Methods Significantly Outperform Default RAG Settings', fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(0, 440)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('/home/user/NeuroEvoRAG/visuals/2_improvement.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 2_improvement.png")


# ── Visual 3: Architecture Diagram ──
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

def draw_box(ax, x, y, w, h, label, color, sublabel=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0), label,
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel,
                ha='center', va='center', fontsize=9, color='#cbd5e1')

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=2))

# Title
ax.text(6, 7.5, 'NeuroEvoRAG Architecture', ha='center', fontsize=18, fontweight='bold', color='white')

# Boxes
draw_box(ax, 0.5, 5.5, 2.5, 1.2, 'HotpotQA', '#475569', 'Dataset')
draw_box(ax, 4, 5.5, 3.5, 1.2, 'RAG Pipeline', '#3b82f6', 'Chunk → Retrieve → Generate')
draw_box(ax, 8.5, 5.5, 3, 1.2, 'Evaluation', '#10b981', 'F1 + Exact Match')

draw_box(ax, 0.5, 3, 2.5, 1.2, 'Evolution', '#8b5cf6', 'Select + Crossover + Mutate')
draw_box(ax, 4, 3, 2.2, 1.2, 'Optuna', '#f59e0b', 'TPE Sampler')
draw_box(ax, 6.8, 3, 2.2, 1.2, 'Grid', '#ef4444', 'Exhaustive')
draw_box(ax, 9.5, 3, 2, 1.2, 'Random', '#06b6d4', 'Uniform')

draw_box(ax, 2, 0.8, 3.5, 1.2, 'Fitness: 0.6*F1 + 0.3*EM', '#1e40af', '+ 0.1*(1-latency)')
draw_box(ax, 7, 0.8, 3.5, 1.2, 'Best Config', '#047857', 'chunk_size, top_k, temp')

# Arrows
draw_arrow(ax, 3, 6.1, 4, 6.1)
draw_arrow(ax, 7.5, 6.1, 8.5, 6.1)
draw_arrow(ax, 10, 5.5, 10, 4.2)

draw_arrow(ax, 1.75, 4.2, 1.75, 5.5)
draw_arrow(ax, 5.1, 4.2, 5.1, 5.5)
draw_arrow(ax, 7.9, 4.2, 7.9, 5.5)
draw_arrow(ax, 10.5, 4.2, 10.5, 5.5)

draw_arrow(ax, 1.75, 3, 3.75, 1.8)
draw_arrow(ax, 5.1, 3, 4.5, 2)
draw_arrow(ax, 7.9, 3, 8.0, 2)
draw_arrow(ax, 10.5, 3, 9.0, 2)

plt.tight_layout()
plt.savefig('/home/user/NeuroEvoRAG/visuals/3_architecture.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 3_architecture.png")


# ── Visual 4: Search Space Discovery ──
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# chunk_size distribution across methods
chunk_data = {
    'Evolution': [2048, 2048, 128, 2048, 128],
    'Optuna': [256, 2048, 1024, 2048, 512],
    'Random': [128, 256, 128, 2048, 128],
}
chunk_positions = [0, 1, 2]
chunk_vals = [[2048, 2048, 128, 2048, 128], [256, 2048, 1024, 2048, 512], [128, 256, 128, 2048, 128]]
bp = axes[0].boxplot(chunk_vals, positions=chunk_positions, patch_artist=True, widths=0.5,
                      boxprops=dict(facecolor=COLORS['evolution'], alpha=0.7),
                      medianprops=dict(color='white', linewidth=2))
for i, color in enumerate([COLORS['evolution'], COLORS['optuna'], COLORS['random']]):
    bp['boxes'][i].set_facecolor(color)
axes[0].set_xticklabels(['Evolution', 'Optuna', 'Random'])
axes[0].set_ylabel('chunk_size')
axes[0].set_title('Chunk Size', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# top_k
k_vals = [[2, 2, 11, 2, 11], [2, 10, 4, 3, 12], [1, 4, 11, 2, 1]]
bp2 = axes[1].boxplot(k_vals, positions=chunk_positions, patch_artist=True, widths=0.5,
                       medianprops=dict(color='white', linewidth=2))
for i, color in enumerate([COLORS['evolution'], COLORS['optuna'], COLORS['random']]):
    bp2['boxes'][i].set_facecolor(color)
axes[1].set_xticklabels(['Evolution', 'Optuna', 'Random'])
axes[1].set_ylabel('top_k')
axes[1].set_title('Retrieval Depth (k)', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# temperature
t_vals = [[0.93, 0.93, 1.14, 0.93, 1.0], [0.18, 0.4, 0.96, 0.82, 1.45], [1.14, 0.3, 1.14, 0.93, 0.23]]
bp3 = axes[2].boxplot(t_vals, positions=chunk_positions, patch_artist=True, widths=0.5,
                       medianprops=dict(color='white', linewidth=2))
for i, color in enumerate([COLORS['evolution'], COLORS['optuna'], COLORS['random']]):
    bp3['boxes'][i].set_facecolor(color)
axes[2].set_xticklabels(['Evolution', 'Optuna', 'Random'])
axes[2].set_ylabel('temperature')
axes[2].set_title('Temperature', fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

fig.suptitle('Parameter Distributions Across Optimization Methods', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/user/NeuroEvoRAG/visuals/4_parameters.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 4_parameters.png")

print("\nAll visuals saved to /home/user/NeuroEvoRAG/visuals/")
