"""
Regression to construct the AIME function.
"""

# Run a regression to predict log AIME
import statsmodels.api as sm
pLvlAll = np.concatenate([MyMarket.agents[j].pLvlHist[:40,:] for j in range(3)],axis=1)
these = np.logical_not(np.isnan(pLvlAll[-1,:]))
pLvlTrim = pLvlAll[:,these]
pLvlSort = np.sort(pLvlTrim,axis=0)
pLvlCapped = np.minimum(pLvlSort[5:,:],10.68)
AIME = np.mean(pLvlCapped,axis=0)
LogAIME = np.log(AIME)
pLvlRet = pLvlTrim[-1,:]
pLogRet = np.log(pLvlRet)
pLogRetSq = pLogRet**2
pLogRetCu = pLogRet**3
Dropout = np.zeros_like(pLogRet)
Dropout[0:MyMarket.agents[0].AgentCount] = 1.
College = np.zeros_like(pLogRet)
College[-MyMarket.agents[2].AgentCount:] = 1.
regressors = np.transpose(np.vstack([np.ones_like(pLogRet),pLogRet,pLogRetSq,pLogRetCu,Dropout,College]))
mod = sm.OLS(LogAIME,regressors)
AIMEresults = mod.fit()
print(AIMEresults.summary(yname='LogAIME',xname=['constant','pLogRet','pLogRetSq','pLogRetCu','Dropout','College']))
