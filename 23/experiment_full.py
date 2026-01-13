import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import os
from tqdm import tqdm
import copy
import inspect

# Configuration
SEEDS = [0, 1, 2, 3, 4]
EPSILONS = [0.1, 0.5, 1.0, 5.0, float('inf')]
BATCH_SIZE = 128 # Larger batch size for faster training on full dataset
EPOCHS = 5      # 10 epochs for full MNIST is enough for reasonable convergence
LR = 0.01
DELTA = 1e-5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
K_TOP = 10      # Top-K for attribution
TEST_SAMPLES_NUM = 100 # Number of test samples to calculate attribution for (to save time)
DATA_ROOT = './data'

print(f"Using device: {DEVICE}")

def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # create a fixed query loader for attribution (subset of full test set)
    # We use a fixed seed for subset selection to ensure comparability across runs if needed, 
    # but here we just take the first N for simplicity or random subset.
    indices = torch.randperm(len(test_dataset), generator=torch.Generator().manual_seed(42))[:TEST_SAMPLES_NUM]
    query_dataset = Subset(test_dataset, indices)
    query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
    
    # Full test loader for accuracy
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Full train loader for attribution (no shuffle, large batch)
    attr_train_loader = DataLoader(train_dataset, batch_size=batch_size*2, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader, query_loader, attr_train_loader


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def get_model():
    # NOTE:
    # Opacus' grad-sample hooks can crash on some models that use in-place residual adds
    # (e.g. `x += shortcut`). A simple CNN avoids that class of issues.
    model = SimpleCNN(num_classes=10)
    return ModuleValidator.fix(model)

def _unwrap_module(model: nn.Module) -> nn.Module:
    return model._module if hasattr(model, "_module") else model


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def train(model, train_loader, epochs, epsilon, device):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = None
    if epsilon != float('inf'):
        privacy_engine = PrivacyEngine()

        # Prefer the exact-epsilon APIs when available.
        if hasattr(privacy_engine, "make_private_with_epsilon"):
            model, optimizer, train_loader = _call_with_supported_kwargs(
                privacy_engine.make_private_with_epsilon,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=epsilon,
                target_delta=DELTA,
                epochs=epochs,
                max_grad_norm=1.0,
                grad_sample_mode="functorch",
            )
        else:
            # Some Opacus versions support target_epsilon in make_private.
            try:
                model, optimizer, train_loader = _call_with_supported_kwargs(
                    privacy_engine.make_private,
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=epsilon,
                    target_delta=DELTA,
                    epochs=epochs,
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                    grad_sample_mode="functorch",
                )
            except TypeError:
                # Fallback: fixed noise multiplier, report realized epsilon.
                model, optimizer, train_loader = _call_with_supported_kwargs(
                    privacy_engine.make_private,
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                    grad_sample_mode="functorch",
                )

    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    realized_epsilon = float('inf')
    if privacy_engine is not None:
        realized_epsilon = privacy_engine.get_epsilon(delta=DELTA)
        print(f"  [DP] Target Epsilon: {epsilon} | Realized Epsilon: {realized_epsilon:.2f}")

    return model, realized_epsilon

def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def compute_gradient_embeddings(model, loader, device, num_samples=None):
    """
    Computes "Last Layer" Gradient Embeddings (Phi * Delta).
    For a linear layer g = x * (p - y).
    We capture 'x' (features) and 'p' (probs) and 'y' (targets).
    """
    model.eval()
    embeddings = []
    deltas = []
    
    # Hook the input to the *last* Linear layer (works for normal and Opacus-wrapped models)
    features = []

    def hook_fn(_module, input, _output):
        features.append(input[0].detach().cpu())

    base = _unwrap_module(model)
    linear_layers = [(name, m) for name, m in base.named_modules() if isinstance(m, nn.Linear)]
    if not linear_layers:
        raise RuntimeError("No nn.Linear layer found to hook for gradient embeddings")
    _name, last_linear = linear_layers[-1]
    handle = last_linear.register_forward_hook(hook_fn)

    count = 0
    with torch.no_grad():
        for data, target in loader:
            features = [] # clear hook buffer
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Probs
            probs = torch.softmax(output, dim=1)
            
            # Construct one-hot target
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, target.view(-1, 1), 1)
            
            # Delta = p - y
            delta = probs - one_hot
            
            # Get feature from hook (it's in the list 'features')
            # features[0] is batch_features
            feat = features[0]
            
            embeddings.append(feat)
            deltas.append(delta.detach().cpu())
            
            count += data.size(0)
            if num_samples and count >= num_samples:
                break
                
    handle.remove()
    return torch.cat(embeddings), torch.cat(deltas)

def get_top_k_influences(params_train, params_test, k=10):
    """
    params_train: (N_train, Dim) - can be tuple (Feats, Deltas)
    params_test: (N_test, Dim)
    
    Influence ~ (Feat_train . Feat_test) * (Delta_train . Delta_test)
    """
    fts_train, dlt_train = params_train
    fts_test, dlt_test = params_test
    
    # Normalize features for stability (cosine sim style) or keep dot product? 
    # Dot product is standard for influence.
    
    # We process in chunks to avoid OOM if N_train is huge
    num_test = fts_test.shape[0]
    top_k_indices = torch.zeros((num_test, k), dtype=torch.long)
    
    # Matrix multiplication: Scores = (F_test @ F_train.T) * (D_test @ D_train.T)
    # This might still be big (N_test x N_train). 100 x 60000 is small (6M floats).
    
    sim_feats = torch.matmul(fts_test, fts_train.t())
    sim_deltas = torch.matmul(dlt_test, dlt_train.t())
    
    scores = sim_feats * sim_deltas # Element-wise
    
    # Get Top-K
    _, indices = torch.topk(scores, k, dim=1)
    return indices

def main():
    if not os.path.exists('results'):
        os.makedirs('results')

    results = []
    
    train_dataset, test_dataset, train_loader, test_loader, query_loader, attr_train_loader = get_dataloaders(BATCH_SIZE)
    
    # Pre-calculate useful info
    print("Data loaded.")

    for seed in SEEDS:
        print(f"\n=== Running Seed {seed} ===")
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Dictionary to store top-k indices for baseline (epsilon = inf)
        baseline_indices = None
        
        # We process 'inf' first to establish baseline
        # Note: We sort EPSILONS to ensure inf comes first if we wanted, but logic handles it.
        sorted_epsilons = sorted(EPSILONS, key=lambda x: -x if x != float('inf') else -float('inf'))
        # Actually inf is typically "baseline", so let's run it.
        
        # To strictly compare "Epsilon X vs Epsilon Inf", we need both models for this seed.
        # But wait, usually we want "Overlap of Model_DP with Model_Base".
        # We need to run the Baseline FIRST.
        
        # Run Baseline (inf)
        print("Training Baseline (No Privacy)...")
        model_base = get_model()
        model_base, _ = train(model_base, train_loader, EPOCHS, float('inf'), DEVICE)
        acc_base = evaluate_accuracy(model_base, test_loader, DEVICE)
        print(f"Baseline Accuracy: {acc_base:.2f}%")
        
        # Compute Baseline Attribution
        print("Computing Baseline Attribution...")
        emb_train_base = compute_gradient_embeddings(model_base, attr_train_loader, DEVICE)
        emb_query_base = compute_gradient_embeddings(model_base, query_loader, DEVICE)
        idx_base = get_top_k_influences(emb_train_base, emb_query_base, k=K_TOP)
        
        del model_base, emb_train_base, emb_query_base # free memory
        
        # Run DP Models
        for eps in EPSILONS:
            if eps == float('inf'):
                # We already did baseline, just record self-overlap (which is 100%) or skip
                # Let's record it as reference
                results.append({
                    'seed': seed,
                    'epsilon': 'Infinity',
                    'accuracy': acc_base,
                    'overlap': 1.0
                })
                continue
                
            print(f"Training DP Model (Target Epsilon {eps})...")
            torch.manual_seed(seed)
            model_dp = get_model()

            # Fresh loader per DP run (Opacus returns a wrapped loader)
            train_loader_fresh = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            model_dp, real_eps = train(model_dp, train_loader_fresh, EPOCHS, eps, DEVICE)

            acc_dp = evaluate_accuracy(model_dp, test_loader, DEVICE)
            print(f"DP Accuracy: {acc_dp:.2f}%, Epsilon: {real_eps:.2f}")
            
            # Attribution
            print("Computing DP Attribution...")
            emb_train_dp = compute_gradient_embeddings(model_dp, attr_train_loader, DEVICE)
            emb_query_dp = compute_gradient_embeddings(model_dp, query_loader, DEVICE)
            idx_dp = get_top_k_influences(emb_train_dp, emb_query_dp, k=K_TOP)
            
            # Calculate Overlap
            # idx_base: (N_query, K)
            # idx_dp: (N_query, K)
            # We compute IoU or simple intersection size / K per query, then average
            overlaps = []
            for i in range(idx_base.shape[0]):
                set_base = set(idx_base[i].tolist())
                set_dp = set(idx_dp[i].tolist())
                intersection = len(set_base.intersection(set_dp))
                overlaps.append(intersection / K_TOP)
            
            avg_overlap = np.mean(overlaps)
            print(f"Overlap: {avg_overlap:.4f}")
            
            results.append({
                'seed': seed,
                'epsilon': eps,
                'accuracy': acc_dp,
                'overlap': avg_overlap
            })
            
            del model_dp, emb_train_dp, emb_query_dp

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv('attribution_results.csv', index=False)
    print("Results saved to attribution_results.csv")
    
    # Plotting
    plot_results(df)

def plot_results(df):
    # Aggregation
    # Treat 'Infinity' as a large number for plotting or separate category
    # We replace 'Infinity' with a placeholder like 10 or 15 for visualization
    
    df_plot = df.copy()
    max_val = df[df['epsilon'] != 'Infinity']['epsilon'].max()
    inf_replacement = max_val * 1.5 if not pd.isna(max_val) else 10.0
    
    df_plot['epsilon_val'] = df_plot['epsilon'].apply(lambda x: inf_replacement if x == 'Infinity' else x)
    
    stats = df_plot.groupby('epsilon_val').agg({
        'overlap': ['mean', 'std', 'count'],
        'accuracy': ['mean', 'std']
    }).reset_index()
    
    stats.columns = ['epsilon', 'overlap_mean', 'overlap_std', 'count', 'acc_mean', 'acc_std']
    
    plt.figure(figsize=(10, 6))
    
    # Plot Overlap
    # Sort by epsilon
    stats = stats.sort_values('epsilon')
    
    x_vals = stats['epsilon']
    y_vals = stats['overlap_mean'] * 100
    y_err = stats['overlap_std'] * 100
    
    # Map back labels
    labels = [str(e) if e != inf_replacement else 'Inf' for e in x_vals]
    
    plt.errorbar(range(len(x_vals)), y_vals, yerr=y_err, fmt='-o', capsize=5, label='Attribution Overlap')
    plt.xticks(range(len(x_vals)), labels)
    
    plt.xlabel(r'Privacy Budget ($\epsilon$)')
    plt.ylabel('Attribution Overlap (%)')
    plt.title('Attribution Fidelity vs Privacy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add Accuracy on secondary axis or just annotation
    for i, row in stats.iterrows():
        plt.annotate(f"Acc: {row['acc_mean']:.1f}%", 
                     (i, row['overlap_mean']*100), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')

    plt.tight_layout()
    plt.savefig('attribution_overlap.png')
    print("Plot saved to attribution_overlap.png")

if __name__ == "__main__":
    main()
