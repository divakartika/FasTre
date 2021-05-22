#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read(#dataset)


# In[ ]:


df.isna().sum()


# In[ ]:


sns.distplot(df[#label])


# In[ ]:


sns.heatmap(df.corr(), annot=True)

