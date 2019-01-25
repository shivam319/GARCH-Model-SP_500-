#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime as dt
import sys
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from arch import arch_model

start = dt.datetime(2015,1,1)
end = dt.datetime(2018,1,1)

sp500 = web.DataReader('SPY','iex', start=start, end=end)

returns = 100 * sp500['close'].pct_change().dropna()
returns.plot()
plt.show()

from arch import arch_model
model=arch_model(returns, vol='Garch', p=1, o=0, q=1, dist='Normal')
results=model.fit()
print(results.summary())

forecasts = results.forecast(horizon=30, method='simulation', simulations=1000)
sims = forecasts.simulations

print(np.percentile(sims.values[-1,:,-1].T,5))
plt.hist(sims.values[-1, :,-1],bins=50)
plt.title('Distribution of Returns')
plt.show()

