"""9 publication-quality figures for AutoML omics pipeline."""
from __future__ import annotations
import logging
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np, pandas as pd, seaborn as sns
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
DPI = 150; TOP_N = 5
MODEL_COLOR = {
    "LogisticL1":"#E74C3C","LogisticL2":"#C0392B","SGD":"#E67E22","LDA":"#F39C12",
    "SVM_linear":"#3498DB","SVM_RBF":"#2980B9","KNN":"#1ABC9C",
    "DecisionTree":"#16A085","RandomForest":"#2ECC71","ExtraTrees":"#27AE60",
    "GradientBoosting":"#9B59B6","AdaBoost":"#8E44AD","NaiveBayes":"#95A5A6",
    "TPOT":"#2C3E50",
}
DS_COLOR = {"balanced":"#3498DB","imbalanced":"#E74C3C","noisy":"#F39C12"}

def _save(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(out_dir)/name; fig.savefig(p,dpi=DPI,bbox_inches="tight"); plt.close(fig)
    logger.info("Saved %s", p)

def fig1_dataset_overview(X,y,otu_names,out_dir):
    pca=PCA(n_components=2); c=pca.fit_transform(X); v=pca.explained_variance_ratio_*100
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    fig.suptitle("Microbiome Dataset Overview (IBD vs Control)",fontsize=13,fontweight="bold")
    ax=axes[0]
    for lbl,col,lab in [(0,"#3498DB","Control"),(1,"#E74C3C","IBD")]:
        idx=y==lbl; ax.scatter(c[idx,0],c[idx,1],c=col,s=25,alpha=0.65,label=lab,edgecolors="white",lw=0.4)
    ax.set_xlabel(f"PC1 ({v[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({v[1]:.1f}%)")
    ax.set_title("PCA — CLR OTU Profiles"); ax.legend(fontsize=9,frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    ax=axes[1]; counts=[(y==0).sum(),(y==1).sum()]
    bars=ax.bar(["Control","IBD"],counts,color=["#3498DB","#E74C3C"],edgecolor="white")
    for b,c2 in zip(bars,counts): ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,str(c2),ha="center",fontsize=11,fontweight="bold")
    ax.set_ylabel("Sample Count"); ax.set_title("Class Distribution"); ax.spines[["top","right"]].set_visible(False)
    ax=axes[2]; vv=X.var(axis=0)
    ax.hist(vv,bins=30,color="#2C3E50",alpha=0.8,edgecolor="white")
    ax.set_xlabel("OTU Variance (CLR)"); ax.set_ylabel("Count")
    ax.set_title(f"OTU Variance Distribution\n(n={len(otu_names)} OTUs)"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig1_dataset_overview.png")

def fig2_leaderboard(bm,tpot_res,out_dir):
    tpot_row=pd.DataFrame([{"model":"TPOT","auc_roc":tpot_res.get("auc_roc",0),"f1":tpot_res.get("f1",0),"mcc":tpot_res.get("mcc",0)}])
    df=pd.concat([bm[["model","auc_roc","f1","mcc"]],tpot_row],ignore_index=True).sort_values("auc_roc",ascending=True)
    fig,ax=plt.subplots(figsize=(10,8))
    colors=[MODEL_COLOR.get(m,"#95A5A6") for m in df["model"]]
    bars=ax.barh(df["model"],df["auc_roc"],color=colors,edgecolor="white")
    for b,v in zip(bars,df["auc_roc"]): ax.text(v+0.002,b.get_y()+b.get_height()/2,f"{v:.3f}",va="center",fontsize=9,fontweight="bold")
    ax.axvline(0.5,color="gray",lw=1,ls="--",alpha=0.5,label="Random")
    ax.set_xlabel("5-Fold CV AUC-ROC",fontsize=11); ax.set_title("AutoML Leaderboard — All Models Ranked",fontsize=13,fontweight="bold")
    ax.set_xlim(0.4,1.08); ax.legend(fontsize=9,frameon=False); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig2_leaderboard.png")

def fig3_performance_grid(bm,out_dir):
    metrics=["auc_roc","auc_pr","f1","accuracy","mcc"]
    mat=bm.set_index("model")[metrics].sort_values("auc_roc",ascending=False)
    mat.index=mat.index.str.replace("_"," ")
    fig,ax=plt.subplots(figsize=(9,8))
    sns.heatmap(mat,annot=True,fmt=".3f",cmap="RdYlGn",vmin=0,vmax=1,ax=ax,linewidths=0.5,
                xticklabels=["AUC-ROC","AUC-PR","F1","Accuracy","MCC"])
    ax.set_yticklabels(ax.get_yticklabels(),fontsize=9)
    ax.set_title("Model Performance Grid",fontsize=13,fontweight="bold",pad=12)
    plt.tight_layout(); _save(fig,out_dir,"fig3_performance_grid.png")

def fig4_roc_curves(automl,X_train,y_train,X_test,y_test,tpot_res,out_dir):
    from sklearn.metrics import roc_curve, roc_auc_score
    from src.models.automl import _has_proba
    fig,ax=plt.subplots(figsize=(8,7)); top5=automl.benchmark_results_.head(TOP_N)["model"].tolist()
    for name in top5:
        pipe=automl._manual_models[name]; pipe.fit(X_train,y_train)
        proba=pipe.predict_proba(X_test)[:,1] if _has_proba(pipe) else pipe.decision_function(X_test)
        if not _has_proba(pipe): proba=(proba-proba.min())/(proba.max()-proba.min()+1e-10)
        fpr,tpr,_=roc_curve(y_test,proba); auc=roc_auc_score(y_test,proba)
        ax.plot(fpr,tpr,color=MODEL_COLOR.get(name,"gray"),lw=1.8,label=f"{name} (AUC={auc:.3f})")
    if tpot_res: ax.plot(tpot_res["fpr"],tpot_res["tpr"],color=MODEL_COLOR["TPOT"],lw=2.5,ls="--",label=f"TPOT (AUC={tpot_res['auc_roc']:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.4); ax.set_xlabel("FPR",fontsize=11); ax.set_ylabel("TPR",fontsize=11)
    ax.set_title("ROC Curves — Top Models + TPOT",fontsize=13,fontweight="bold")
    ax.legend(fontsize=8,frameon=False,loc="lower right"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig4_roc_curves.png")

def fig5_pr_curves(automl,X_train,y_train,X_test,y_test,tpot_res,out_dir):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from src.models.automl import _has_proba
    fig,ax=plt.subplots(figsize=(8,7)); top5=automl.benchmark_results_.head(TOP_N)["model"].tolist()
    for name in top5:
        pipe=automl._manual_models[name]; pipe.fit(X_train,y_train)
        proba=pipe.predict_proba(X_test)[:,1] if _has_proba(pipe) else pipe.decision_function(X_test)
        if not _has_proba(pipe): proba=(proba-proba.min())/(proba.max()-proba.min()+1e-10)
        p,r,_=precision_recall_curve(y_test,proba); ap=average_precision_score(y_test,proba)
        ax.plot(r,p,color=MODEL_COLOR.get(name,"gray"),lw=1.8,label=f"{name} (AP={ap:.3f})")
    if tpot_res: ax.plot(tpot_res["rec_curve"],tpot_res["prec_curve"],color=MODEL_COLOR["TPOT"],lw=2.5,ls="--",label=f"TPOT (AP={tpot_res['auc_pr']:.3f})")
    ax.set_xlabel("Recall",fontsize=11); ax.set_ylabel("Precision",fontsize=11)
    ax.set_title("Precision-Recall Curves",fontsize=13,fontweight="bold")
    ax.legend(fontsize=8,frameon=False); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig5_pr_curves.png")

def fig6_cv_boxplot(cv_details,out_dir):
    names=sorted(cv_details,key=lambda m: np.mean(cv_details[m]),reverse=True)
    data=[cv_details[m] for m in names]; colors=[MODEL_COLOR.get(m,"#95A5A6") for m in names]
    fig,ax=plt.subplots(figsize=(13,5))
    bp=ax.boxplot(data,patch_artist=True,notch=False,medianprops=dict(color="black",linewidth=2))
    for patch,col in zip(bp["boxes"],colors): patch.set_facecolor(col); patch.set_alpha(0.75)
    for i,(d,col) in enumerate(zip(data,colors),start=1):
        rng=np.random.default_rng(i); j=rng.uniform(-0.15,0.15,len(d))
        ax.scatter(np.full(len(d),i)+j,d,c=col,s=30,zorder=3,alpha=0.85,edgecolors="white")
    ax.set_xticks(range(1,len(names)+1)); ax.set_xticklabels(names,rotation=35,ha="right",fontsize=9)
    ax.set_ylabel("AUC-ROC (5-fold CV)",fontsize=11); ax.set_title("Cross-Validation Stability",fontsize=13,fontweight="bold")
    ax.axhline(0.5,color="gray",lw=1,ls="--",alpha=0.5); ax.set_ylim(0.3,1.05); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig6_cv_boxplot.png")

def fig7_feature_importance(feat_imp,out_dir,n_top=20):
    top=feat_imp.nlargest(n_top); cols=[c.replace("_"," ") for c in top.index]
    norm=top.values/top.values.max(); colors=plt.cm.RdYlGn(norm)
    fig,ax=plt.subplots(figsize=(9,7))
    ax.barh(range(n_top),top.values[::-1],color=colors[::-1],edgecolor="white")
    ax.set_yticks(range(n_top)); ax.set_yticklabels(cols[::-1],fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)",fontsize=11)
    ax.set_title(f"Top {n_top} Discriminative OTUs\n(Best Tree-Based Model)",fontsize=12,fontweight="bold")
    ax.spines[["top","right"]].set_visible(False); plt.tight_layout(); _save(fig,out_dir,"fig7_feature_importance.png")

def fig8_dataset_robustness(rob_df,out_dir):
    metrics=["auc_roc","auc_pr","f1"]; labels=["AUC-ROC","AUC-PR","F1"]
    ds=rob_df["dataset"].tolist(); x=np.arange(len(ds)); w=0.25
    fig,ax=plt.subplots(figsize=(9,5))
    for j,(m,lbl) in enumerate(zip(metrics,labels)):
        vals=rob_df[m].tolist()
        bars=ax.bar(x+j*w,vals,w,label=lbl,color=[DS_COLOR.get(d,"gray") for d in ds],alpha=0.8,edgecolor="white")
        for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{v:.3f}",ha="center",fontsize=8,fontweight="bold")
    ax.set_xticks(x+w); ax.set_xticklabels([d.capitalize() for d in ds],fontsize=11)
    ax.set_ylim(0,1.15); ax.set_ylabel("Score",fontsize=11)
    ax.set_title("AutoML Robustness: Balanced vs Imbalanced vs Noisy",fontsize=12,fontweight="bold")
    ax.legend(fontsize=9,frameon=False); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig8_dataset_robustness.png")

def fig9_summary(automl,best_manual,tpot_res,out_dir):
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    fig.suptitle("AutoML Pipeline Summary — Microbiome IBD Classification",fontsize=13,fontweight="bold")
    ax=axes[0]; names=[best_manual["model"],"TPOT"]; aucs=[best_manual["auc_roc"],tpot_res.get("auc_roc",0)]
    colors=[MODEL_COLOR.get(names[0],"#3498DB"),MODEL_COLOR["TPOT"]]
    bars=ax.bar(names,aucs,color=colors,edgecolor="white",width=0.5)
    for b,v in zip(bars,aucs): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.005,f"{v:.3f}",ha="center",fontsize=12,fontweight="bold")
    ax.set_ylim(0,1.12); ax.set_ylabel("Test AUC-ROC"); ax.set_title("Best Manual vs TPOT"); ax.spines[["top","right"]].set_visible(False)
    ax=axes[1]; cm=best_manual["confusion"]
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax,xticklabels=["Ctrl","IBD"],yticklabels=["Ctrl","IBD"],cbar=False,linewidths=0.5)
    ax.set_title(f"Confusion Matrix\n({best_manual['model']})",fontsize=11); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax=axes[2]; bm=automl.benchmark_results_
    for _,row in bm.iterrows():
        col=MODEL_COLOR.get(row["model"],"gray")
        ax.scatter(row["time_s"],row["auc_roc"],c=col,s=80,edgecolors="white",lw=0.5,zorder=3)
        ax.annotate(row["model"],(row["time_s"],row["auc_roc"]),fontsize=6,ha="left",va="bottom",xytext=(2,2),textcoords="offset points")
    ax.set_xlabel("Training Time (s, 5-fold CV)",fontsize=10); ax.set_ylabel("AUC-ROC",fontsize=10)
    ax.set_title("Speed vs Performance",fontsize=11); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); _save(fig,out_dir,"fig9_summary.png")

def generate_all(X,y,X_train,y_train,X_test,y_test,automl,tpot_res,best_manual,robustness_df,otu_names,out_dir):
    out_dir=Path(out_dir); logger.info("Generating figures → %s", out_dir)
    fig1_dataset_overview(X,y,otu_names,out_dir)
    fig2_leaderboard(automl.benchmark_results_,tpot_res,out_dir)
    fig3_performance_grid(automl.benchmark_results_,out_dir)
    fig4_roc_curves(automl,X_train,y_train,X_test,y_test,tpot_res,out_dir)
    fig5_pr_curves(automl,X_train,y_train,X_test,y_test,tpot_res,out_dir)
    fig6_cv_boxplot(automl.cv_details_,out_dir)
    if automl.feature_importance_ is not None:
        fig7_feature_importance(automl.feature_importance_,out_dir)
    fig8_dataset_robustness(robustness_df,out_dir)
    fig9_summary(automl,best_manual,tpot_res,out_dir)
    logger.info("All 9 figures saved.")
