"""Simulate microbiome OTU data for AutoML benchmarking."""
from __future__ import annotations
import numpy as np
import pandas as pd

FIRMICUTES = ["Faecalibacterium_prausnitzii","Roseburia_intestinalis","Blautia_obeum","Eubacterium_rectale","Eubacterium_hallii","Ruminococcus_bromii","Coprococcus_eutactus","Lachnospira_multipara","Anaerostipes_caccae","Subdoligranulum_variabile","Dorea_longicatena","Clostridium_leptum"]
BACTEROIDETES = ["Bacteroides_thetaiotaomicron","Bacteroides_vulgatus","Bacteroides_uniformis","Prevotella_copri","Alistipes_shahii","Parabacteroides_distasonis","Phocaeicola_dorei","Alistipes_putredinis"]
ACTINOBACTERIA = ["Bifidobacterium_longum","Bifidobacterium_adolescentis","Collinsella_aerofaciens","Eggerthella_lenta"]
IBD_UP = ["Escherichia_coli","Klebsiella_pneumoniae","Fusobacterium_nucleatum","Ruminococcus_gnavus","Peptostreptococcus_stomatis"]
IBD_DOWN = ["Akkermansia_muciniphila","Bifidobacterium_bifidum","Lactobacillus_rhamnosus","Christensenellaceae_sp","Oscillospira_sp"]

def _otu_names(n=150):
    named = FIRMICUTES+BACTEROIDETES+ACTINOBACTERIA+IBD_UP+IBD_DOWN
    return (named+[f"OTU_{i:04d}" for i in range(n-len(named))])[:n]

OTU_NAMES = _otu_names(150)
N_OTUS = len(OTU_NAMES)
CTRL_IDX = [OTU_NAMES.index(t) for t in FIRMICUTES+BACTEROIDETES+IBD_DOWN if t in OTU_NAMES]
IBD_IDX  = [OTU_NAMES.index(t) for t in IBD_UP if t in OTU_NAMES]

def _clr(X, eps=1e-6):
    X = X+eps; lx=np.log(X); return lx-lx.mean(axis=1, keepdims=True)

def _make_alpha(label):
    a = np.ones(N_OTUS)*0.25
    for i in CTRL_IDX: a[i]=2.0
    for i in IBD_IDX:  a[i]=0.2
    if label==1:
        for i in CTRL_IDX: a[i]=max(0.3,a[i]*0.4)
        for i in IBD_IDX:  a[i]=3.5
    return a

def simulate_dataset(n_samples=400, ibd_fraction=0.50, label_noise=0.0, seed=42):
    rng=np.random.default_rng(seed)
    n_ibd=int(n_samples*ibd_fraction); n_ctrl=n_samples-n_ibd
    rows,labels=[],[]
    for label,n in [(0,n_ctrl),(1,n_ibd)]:
        for _ in range(n):
            raw=rng.dirichlet(_make_alpha(label))
            mask=rng.random(N_OTUS)<0.38; raw[mask]=0
            raw/=raw.sum()+1e-10; rows.append(raw); labels.append(label)
    clr=_clr(np.array(rows))
    ids=[f"CTRL_{i+1:04d}" for i in range(n_ctrl)]+[f"IBD_{i+1:04d}" for i in range(n_ibd)]
    y=np.array(labels,dtype=int)
    if label_noise>0:
        fi=rng.choice(len(y),int(label_noise*len(y)),replace=False); y[fi]=1-y[fi]
    return pd.DataFrame(clr,columns=OTU_NAMES,index=ids), pd.Series(y,index=ids,name="label")

def get_all_datasets(seed=42):
    return {
        "balanced":   simulate_dataset(400,0.50,0.00,seed),
        "imbalanced": simulate_dataset(400,0.25,0.00,seed+1),
        "noisy":      simulate_dataset(400,0.50,0.20,seed+2),
    }
