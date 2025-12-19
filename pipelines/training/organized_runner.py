"""
Enhanced runner with organized output folders per run
Mỗi lần chạy sẽ có folder riêng chứa tất cả outputs
"""
from pathlib import Path
from datetime import datetime
import json
import shutil

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_ROOT = BASE_DIR / "results"

def create_run_folder(dataset_name, model_name, attack_name="none"):
    """
    Tạo folder cho mỗi lần chạy với cấu trúc rõ ràng
    
    Structure:
    results/
    └── runs/
        └── 20251216_101530_cicids2017_mlp_gan/
            ├── logs/
            │   └── run_info.json
            ├── metrics/
            │   ├── baseline_metrics.csv
            │   ├── classification_report.csv
            │   ├── confusion_matrix.csv
            │   └── gan_adversarial_metrics.csv
            ├── models/
            │   └── mlp_model.pkl
            └── adversarial/
                ├── adversarial_samples.npy
                └── perturbation_stats.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{dataset_name}_{model_name}_{attack_name}"
    
    run_dir = RESULTS_ROOT / "runs" / folder_name
    
    # Create subfolders
    folders = {
        'root': run_dir,
        'logs': run_dir / 'logs',
        'metrics': run_dir / 'metrics',
        'models': run_dir / 'models',
        'adversarial': run_dir / 'adversarial',
    }
    
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme = run_dir / 'README.md'
    readme_content = f"""# Run: {folder_name}

**Created**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration
- Dataset: {dataset_name}
- Model: {model_name}
- Attack: {attack_name}

## Folder Structure
- `logs/` - JSON logs and run information
- `metrics/` - All metric CSV files
- `models/` - Trained model files
- `adversarial/` - Adversarial samples and attack results

## Files
(Will be populated after run completes)
"""
    readme.write_text(readme_content)
    
    return folders, timestamp

def save_run_summary(run_folder, summary_data):
    """Save comprehensive run summary."""
    summary_file = run_folder['logs'] / 'run_info.json'
    summary_file.write_text(json.dumps(summary_data, indent=2, default=str))
    
    # Update README with actual files
    readme = run_folder['root'] / 'README.md'
    current_readme = readme.read_text()
    
    files_list = "\n### Generated Files:\n\n"
    for folder_name, folder_path in run_folder.items():
        if folder_name == 'root':
            continue
        files = list(folder_path.glob('*'))
        if files:
            files_list += f"\n**{folder_name}/:**\n"
            for f in files:
                size = f.stat().st_size / 1024  # KB
                files_list += f"- `{f.name}` ({size:.1f} KB)\n"
    
    readme.write_text(current_readme + files_list)

def create_run_index():
    """Tạo file index.md liệt kê tất cả các runs."""
    runs_dir = RESULTS_ROOT / "runs"
    if not runs_dir.exists():
        return
    
    runs = sorted(runs_dir.glob("*"), reverse=True)  # Newest first
    
    index_content = f"""# Experiment Runs Index

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Runs**: {len(runs)}

---

## Recent Runs

"""
    
    for run_dir in runs[:20]:  # Show last 20 runs
        if not run_dir.is_dir():
            continue
        
        # Parse folder name: timestamp_dataset_model_attack
        parts = run_dir.name.split('_')
        if len(parts) >= 4:
            timestamp = f"{parts[0]}_{parts[1]}"
            dataset = parts[2]
            model = parts[3]
            attack = '_'.join(parts[4:]) if len(parts) > 4 else 'none'
            
            # Try to read metrics
            metrics_info = ""
            run_info_file = run_dir / 'logs' / 'run_info.json'
            if run_info_file.exists():
                try:
                    with open(run_info_file) as f:
                        data = json.load(f)
                    model_acc = data.get('model', {}).get('outputs', {}).get('accuracy', 'N/A')
                    if isinstance(model_acc, (int, float)):
                        metrics_info = f" - Accuracy: {model_acc:.4f}"
                except:
                    pass
            
            index_content += f"### [{run_dir.name}](./{run_dir.name}/)\n"
            index_content += f"- **Dataset**: {dataset}\n"
            index_content += f"- **Model**: {model}\n"
            index_content += f"- **Attack**: {attack}\n"
            index_content += f"- **Time**: {timestamp}{metrics_info}\n"
            index_content += f"- **Path**: `{run_dir.relative_to(BASE_DIR)}/`\n\n"
    
    index_file = runs_dir / "INDEX.md"
    index_file.write_text(index_content)
    print(f"\n📚 Run index created: {index_file}")

# Example usage function
def example_usage():
    """
    Example how to use in training code:
    
    from pipelines.training.organized_runner import create_run_folder, save_run_summary
    
    # At start of training
    folders, timestamp = create_run_folder('cicids2017', 'mlp', 'gan')
    
    # During training, save to organized locations:
    # - metrics → folders['metrics'] / 'baseline_metrics.csv'
    # - models → folders['models'] / 'mlp_model.pkl'
    # - adversarial → folders['adversarial'] / 'samples.npy'
    
    # At end, save summary
    summary = {
        'run_id': timestamp,
        'model': {'accuracy': 0.995},
        # ... other info
    }
    save_run_summary(folders, summary)
    create_run_index()
    """
    pass

__all__ = ['create_run_folder', 'save_run_summary', 'create_run_index']
