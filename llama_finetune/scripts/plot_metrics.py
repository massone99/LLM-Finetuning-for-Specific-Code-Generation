import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Load the JSON data
with open('../res/data/trained_models/trained_models.json', 'r') as f:
    models = json.load(f)

# Group models by training set size
size_groups = defaultdict(list)
for m in models:
    size_groups[m['train_dataset_size']].append(m)

# Find best models for each size
best_models = []
for size, group in size_groups.items():
    # Find model with best BLEU score in this group
    best_bleu = max(group, key=lambda m: m['avg_bleu'])
    # Find model with best success rate in this group
    best_success = max(group, key=lambda m: m['execution_check']['successful_runs'] / m['execution_check']['total_snippets'])
    best_models.extend([best_bleu, best_success])

# Extract sorted unique data points
train_sizes = sorted(list(set(m['train_dataset_size'] for m in best_models)))
best_bleu_scores = [max(m['avg_bleu'] for m in best_models if m['train_dataset_size'] == size) for size in train_sizes]
best_success_rates = [max(m['execution_check']['successful_runs'] / m['execution_check']['total_snippets'] 
                         for m in best_models if m['train_dataset_size'] == size) for size in train_sizes]

# Create and save BLEU score plot
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, best_bleu_scores, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Training Set Size')
plt.ylabel('Best Average BLEU Score')
plt.title('Best BLEU Score vs Training Set Size')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../res/plots/bleu_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# Create and save success rate plot
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, best_success_rates, 'orange', marker='o', linewidth=2, markersize=8)
plt.xlabel('Training Set Size')
plt.ylabel('Best Success Rate')
plt.title('Best Success Rate vs Training Set Size')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../res/plots/success_rates.png', dpi=300, bbox_inches='tight')
plt.close()
