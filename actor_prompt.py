# The basic prompt for the LLM
import pandas as pd

df = pd.read_csv("...")
TARGET_NAME = ""
clf_flg = df[TARGET_NAME].nunique() > 4
data_desc = "No desc"

PYTHON_FORMAT_BLOCK = \
"""
# Name_Alex_flg: Description
# Usefulness: Rationale for the usefulness of a feature
# Input samples: 'Name': ["Alex", "Mary", "Jonny"]
df['Name_Alex_flg'] = df['Name'] == "Alex"
"""

data_info = "info from df.info()"

prompt = f"""\
The dataset is loaded into the ’df’ variable in pandas.DataFrame format and is available for any manipulation.
All information is taken from the dataset using the pandas library. Information about the number of omissions and
column types (data.info()):
{data_info} ’df.sample()’ (data sample): {df.sample()} ‘df.corr()‘ (correlation for num. features): {df.corr()}
’Useful information about the data:’ { data_desc }
’df.describe() (statistics in the features )’
{ df.describe()} This code generates additional columns based on data information, feature names, and
other useful information.
The code is posted as the best example of feature generation by LLM with extensive experience in researching data and
creating useful features The generated features are useful for solving the
{'classification' if clf_flg else 'regression'} problem using gradient boosting algorithm LightGBM
(therefore, feature generation based on scale changes or feature combinations do not make sense).
The target variable in the data is ’{TARGET_NAME}’, the quality metric is {'ROC-AUC' if clf_flg else 'RMSE'}.
The generated features bring new logical information to the real-world data, useful for solving the problem.
Some approaches used for generation:
*. Type transformations. For example: from a numeric feature to make several categorical features
*. Creating flags. For example: putting a true flag when some conditions are fulfilled in one record,
*. Discretization. For example: divide a numerical feature into intervals and assign each interval a number
*. Complete deletion of the feature. In the case of a small number of records, it may help not to overfit
*. Changing the feature values according to the condition.
For example: replacing erroneous values with the most appropriate ones.
*. Any useful transformation based on knowledge about the real world
The information is generated in [DOMAIN]: a desc of useful information about the data and [FEATURES]:
where the python blocks begin.
Feature generation example: {PYTHON_FORMAT_BLOCK}
Each block was checked for possible errors: the absence of a feature in the table and the correctness of the syntax
[LLM ANSWER] Following the proposed work format, first [DOMAIN] describing all the necessary information for
the data work is written out, followed by [FEATURES]: with the generated features in Python block format.
[DOMAIN]:
"""
