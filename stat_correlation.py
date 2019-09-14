import numpy as np
import scipy.stats as stats

n = 50
x = np.random.normal(size=n)
y = 2 *x + np.random.normal(size=n)

#compute pearson R with scipy
cor, pval = stats.pearsonr(x,y)
print(cor, pval) 
