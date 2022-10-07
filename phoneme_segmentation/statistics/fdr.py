import numpy as np

def fdr_qvalues(pvalues, lam=0.5):
    m = float(pvalues.size)
    psinds = np.argsort(pvalues)
    psoriginds = np.argsort(psinds)
    sortpvals = pvalues[psinds]
    pi_0 = max((pvalues > lam).sum(),1) / ((1-lam) * m)
    print (pi_0)
    p_less = (np.arange(m)+1) / m
    pfdr = pi_0 * sortpvals / (p_less * (1 - (1-sortpvals)**m))
    print (pfdr)
    qvals = np.zeros((m,))
    qvals[m-1] = pfdr[m-1]
    for ii in range(2, int(m)+1):
        qvals[m-ii] = min(pfdr[m-ii], qvals[m-ii+1])

    return qvals[psoriginds]

def fdr_correct(pval, thres):
   """Find the fdr corrected p-value thresholds
   pval - vector of p-values
   thres - FDR level
   pID - p-value thres based on independence or positive dependence
   pN - Nonparametric p-val thres"""
   # remove NaNs
   p = pval[np.nonzero(np.isnan(pval)==False)[0]]
   p = np.sort(p)
   V = np.float(len(p))
   I = np.arange(V) + 1

   cVID = 1
   cVN = (1/I).sum()

   th1 = np.nonzero(p <= I/V*thres/cVID)[0]
   th2 = np.nonzero(p <= I/V*thres/cVN)[0]
   if len(th1)>0:
       pID = p[th1.max()]
   else:
       pID = -np.inf
   if len(th2)>0:
       pN =  p[th2.max()]
   else:
       pN = -np.inf

   return pID, pN
