#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Resampling Techniques

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter


# # Read the CSV and Perform Basic Data Cleaning

# In[3]:


columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]

target = ["loan_status"]


# In[4]:


# Load the data
file_path = Path('LoanStats_2019Q1.csv')
df = pd.read_csv(file_path, skiprows=1)[:-2]
df = df.loc[:, columns].copy()

# Drop the null columns where all values are null
df = df.dropna(axis='columns', how='all')

# Drop the null rows
df = df.dropna()

# Remove the `Issued` loan status
issued_mask = df['loan_status'] != 'Issued'
df = df.loc[issued_mask]

# convert interest rate to numerical
df['int_rate'] = df['int_rate'].str.replace('%', '')
df['int_rate'] = df['int_rate'].astype('float') / 100


# Convert the target column values to low_risk and high_risk based on their values
x = {'Current': 'low_risk'}   
df = df.replace(x)

x = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'high_risk')    
df = df.replace(x)

df.reset_index(inplace=True, drop=True)

df.head()


# # Split the Data into Training and Testing

# In[ ]:





# In[14]:


# Create our features
X = df.values.reshape(-1, 1)


# Create our target
y = df


# In[6]:


X.describe()


# In[7]:


# Check the balance of our target values
y['loan_status'].value_counts()


# In[13]:


from sklearn.model_selection import train_test_split
X, y = train_test_split(X,
    y, random_state=1, stratify=y)


# # Oversampling
# 
# In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:
# 
# 1. View the count of the target classes using `Counter` from the collections library. 
# 3. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Print the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# ### Naive Random Oversampling

# In[9]:


# Resample the training data with the RandomOversampler
# YOUR CODE HERE


# In[10]:


# Train the Logistic Regression model using the resampled data
# YOUR CODE HERE


# In[11]:


# Calculated the balanced accuracy score
# YOUR CODE HERE


# In[12]:


# Display the confusion matrix
# YOUR CODE HERE


# In[13]:


# Print the imbalanced classification report
# YOUR CODE HERE


# ### SMOTE Oversampling

# In[14]:


# Resample the training data with SMOTE
# YOUR CODE HERE


# In[15]:


# Train the Logistic Regression model using the resampled data
# YOUR CODE HERE


# In[16]:


# Calculated the balanced accuracy score
# YOUR CODE HERE


# In[17]:


# Display the confusion matrix
# YOUR CODE HERE


# In[18]:


# Print the imbalanced classification report
# YOUR CODE HERE


# # Undersampling
# 
# In this section, you will test an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:
# 
# 1. View the count of the target classes using `Counter` from the collections library. 
# 3. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Print the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# In[19]:


# Resample the data using the ClusterCentroids resampler
# Warning: This is a large dataset, and this step may take some time to complete
# YOUR CODE HERE


# In[20]:


# Train the Logistic Regression model using the resampled data
# YOUR CODE HERE


# In[21]:


# Calculated the balanced accuracy score
# YOUR CODE HERE


# In[22]:


# Display the confusion matrix
# YOUR CODE HERE


# In[23]:


# Print the imbalanced classification report
# YOUR CODE HERE


# # Combination (Over and Under) Sampling
# 
# In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. You will resample the data using the SMOTEENN algorithm and complete the folliowing steps:
# 
# 1. View the count of the target classes using `Counter` from the collections library. 
# 3. Use the resampled data to train a logistic regression model.
# 3. Calculate the balanced accuracy score from sklearn.metrics.
# 4. Print the confusion matrix from sklearn.metrics.
# 5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
# 
# Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

# In[24]:


# Resample the training data with SMOTEENN
# Warning: This is a large dataset, and this step may take some time to complete
# YOUR CODE HERE


# In[25]:


# Train the Logistic Regression model using the resampled data
# YOUR CODE HERE


# In[26]:


# Calculated the balanced accuracy score
# YOUR CODE HERE


# In[27]:


# Display the confusion matrix
# YOUR CODE HERE


# In[28]:


# Print the imbalanced classification report
# YOUR CODE HERE


# In[ ]:




