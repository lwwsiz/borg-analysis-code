"""Existing RNA-ATAC HC3 regression and BH adjustment functions.

Function bodies are copied unchanged from the analysis source. Donor selection,
program projection and input assembly are outside this selected source module.
The manuscript specifies the primary and depth-adjusted designs.
"""
import numpy as np
from scipy import stats

def fit(x,y,covs=None):
 x,y=np.array(x,float),np.array(y,float);cols=[np.ones(len(x)),x]+([] if covs is None else [np.asarray(c,float) for c in covs]);X=np.column_stack(cols);rank=np.linalg.matrix_rank(X);df=len(x)-X.shape[1]
 if rank<X.shape[1] or df<=0:return {'status':'NOT_IDENTIFIABLE','n':len(x),'rank':int(rank),'columns':X.shape[1],'df':df}
 inv=np.linalg.inv(X.T@X);beta=inv@X.T@y;res=y-X@beta;h=np.sum((X@inv)*X,axis=1)
 if h.max()>=1-1e-8:return {'status':'HC3_NOT_DEFINED','n':len(x),'max_leverage':float(h.max())}
 V=inv@(X.T@((res/(1-h))[:,None]**2*X))@inv;se=np.sqrt(V[1,1]);crit=stats.t.ppf(.975,df)
 return {'status':'ESTIMATED','n':len(x),'slope':float(beta[1]),'HC3_se':float(se),'ci95':[float(beta[1]-crit*se),float(beta[1]+crit*se)],'p':float(2*stats.t.sf(abs(beta[1]/se),df)),'df':df,'max_leverage':float(h.max()),'spearman_rho_unadjusted':float(stats.spearmanr(x,y).statistic),'RNA_sd':float(y.std(ddof=1)),'ATAC_sd':float(x.std(ddof=1))}

def bh(items):
 order=sorted(range(len(items)),key=lambda i:items[i]['p']);q=1.
 for rank in range(len(order),0,-1):i=order[rank-1];q=min(q,items[i]['p']*len(order)/rank);items[i]['BH_q']=float(q)
