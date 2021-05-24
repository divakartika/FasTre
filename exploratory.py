#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('WaitDataF1.csv')
df = df.sort_index(axis=0,ascending=False,ignore_index=True)


# In[ ]:


print(df.isna().sum())


# In[ ]:


sns.distplot(df['Wait'])


# In[ ]:
    
correlation = df.corr()

# In[ ]:
    
desc = df.describe()

# In[ ]:


sns.heatmap(df.corr(), annot=True)

