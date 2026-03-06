"""Day 24 · AutoML Pipeline for Omics Data — entry point."""
from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data.simulator      import simulate_dataset, get_all_datasets, OTU_NAMES
from src.models.automl       import AutoMLPipeline
from src.visualization.plots import generate_all
from src.utils.logger        import setup_logging
from src.utils.config        import load_config

logger = logging.getLogger(__name__)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml")
    p.add_argument("--outdir",default="outputs"); p.add_argument("--quiet",action="store_true")
    return p.parse_args()

def main():
    args=parse_args(); cfg=load_config(args.config)
    setup_logging(level=logging.WARNING if args.quiet else logging.INFO)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    t0=time.time()
    logger.info("="*60); logger.info("Day 24 · AutoML Pipeline for Omics Data"); logger.info("="*60)
    data_cfg=cfg.get("data",{}); ml_cfg=cfg.get("automl",{})

    logger.info("[1/6] Simulating microbiome dataset …")
    X_df,y_s=simulate_dataset(n_samples=data_cfg.get("n_samples",400),
        ibd_fraction=data_cfg.get("ibd_fraction",0.5),
        label_noise=data_cfg.get("label_noise",0.0),seed=data_cfg.get("seed",42))
    X=X_df.values; y=y_s.values
    logger.info("Dataset: %d×%d  IBD=%d  Ctrl=%d",*X.shape,y.sum(),(y==0).sum())
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=data_cfg.get("test_size",0.20),
        stratify=y,random_state=data_cfg.get("seed",42))
    logger.info("Train: %d | Test: %d",len(X_tr),len(X_te))

    logger.info("[2/6] Running 13-model benchmark …")
    automl=AutoMLPipeline(seed=ml_cfg.get("seed",42),cv_folds=ml_cfg.get("cv_folds",5),
        flaml_budget=ml_cfg.get("flaml_budget",60))
    bm=automl.run_benchmark(X_tr,y_tr,OTU_NAMES)
    logger.info("Benchmark done. Top: %s  AUC=%.3f",bm.iloc[0]["model"],bm.iloc[0]["auc_roc"])

    logger.info("[3/6] Running FLAML AutoML …")
    flaml_res=automl.run_automl(X_tr,y_tr,X_te,y_te)

    logger.info("[4/6] Evaluating best manual model …")
    best_manual=automl.evaluate_best_manual(X_tr,y_tr,X_te,y_te)

    logger.info("[5/6] Robustness check …")
    all_ds={k:(df.values,s.values) for k,(df,s) in get_all_datasets(seed=data_cfg.get("seed",42)).items()}
    rob_df=automl.benchmark_datasets(all_ds,model_name="RandomForest")

    logger.info("[6/6] Saving outputs and figures …")
    (out/"models").mkdir(exist_ok=True)
    automl.save(out/"models"/"automl_pipeline.joblib")
    bm.to_csv(out/"benchmark_results.csv",index=False)
    rob_df.to_csv(out/"robustness_results.csv",index=False)
    pd.DataFrame({m:automl.cv_details_[m] for m in automl.cv_details_}).to_csv(out/"cv_fold_aucs.csv",index=False)
    if automl.feature_importance_ is not None:
        automl.feature_importance_.head(30).to_csv(out/"top_features.csv",header=["importance"])
    generate_all(X,y,X_tr,y_tr,X_te,y_te,automl,flaml_res,best_manual,rob_df,OTU_NAMES,out)

    elapsed=time.time()-t0
    print("\n"+"="*58)
    print("  Day 24 · AutoML Pipeline Summary")
    print("="*58)
    print(f"  Samples    : {len(X_df)} ({len(X_tr)} train / {len(X_te)} test)")
    print(f"  Features   : {X_df.shape[1]} OTUs (CLR)")
    print(f"  Models     : {len(bm)} manual + FLAML AutoML")
    print()
    print("  Leaderboard (5-fold CV AUC-ROC):")
    for _,row in bm.head(5).iterrows():
        print(f"    {row['model']:<22}  AUC={row['auc_roc']:.3f}  F1={row['f1']:.3f}")
    print(f"    {'FLAML AutoML (test)':<22}  AUC={flaml_res['auc_roc']:.3f}  F1={flaml_res['f1']:.3f}  [{flaml_res['best_estimator']}]")
    print()
    print("  Robustness (Random Forest):")
    for _,row in rob_df.iterrows():
        print(f"    {row['dataset']:<14}  AUC={row['auc_roc']:.3f}  F1={row['f1']:.3f}")
    print(f"\n  Figures    : 9 saved to {out}/")
    print(f"  Elapsed    : {elapsed:.1f}s")
    print("="*58+"\n")

if __name__=="__main__": main()
