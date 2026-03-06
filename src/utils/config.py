from pathlib import Path; import yaml
def load_config(p):
    p = Path(p)
    if not p.exists(): return {}
    with open(p) as f: return yaml.safe_load(f) or {}
