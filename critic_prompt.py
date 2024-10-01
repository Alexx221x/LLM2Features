
TARGET_NAME = ""
# Can also be determined automatically by the number of unique values in the target set
TASK_NAME = "classification" if True else "regression"
INIT_PROMPT = ""
DUMMY_PROMPT = ""
FIRST_LLM_ANSWER = ""
prompt = \
f"""
You are a LLM-critic that receives as input the output of another LLM model.
You need to fix syntax and logic errors in python code in order to improve training quality and prediction metrics
on the {TARGET_NAME} feature when training LightGBM on the {TASK_NAME} task.
The [INIT PROMPT] will be passed first, which is the OTHER LLM prompt that was used to generate the features.
Next, the OTHER LLM response will be transmitted after the < ACTOR LLM ANSWER > token.
Generating start after <LLM CRITIC ANSWER>.
You should write strictly in [ERROR DESCRIPTION] format all errors that were made in the traits,
including in the calculation logic or with leaked test data from the target [FIX] python code in blocks.
Example:
[INIT_PROMPT]
{DUMMY_PROMPT}

< ACTOR LLM ANSWER >
[DOMAIN] Count features 1 + 1 and remove features df[’Name’] because this will come in handy
for predicting the target.
[FEATURES]
“‘python
Feature: Adding two integer features
Usefulness: This feature is needed for predicting the target.
Input samples: ’Number 1’: [1, 0, 3], ’Number 2’: [0, 2, 1], ’Number 3’: [4, 5, -1]
df[’Sum Number 1 and 2’] = df[’Number 1’] + df[’Number 2’] + df[’Number 3’]
“‘
“‘python
Feature: Removing the ’Name’ feature
Usefulness: This feature does not affect targeting
df.drop(columns=[’Name’])
“‘
<LLM CRITIC ANSWER>
[ERROR DESCRIPTION]
The feature ’Sum Number 1 and 2’ should consist of the sums of the two
integer columns ’Number 1’ and ’Number 2’. Since ’Number 3’ is added to it, this sign contains an error
The ’Name’ sign was not deleted because the inplace=True argument is missing.
[FIX]
“‘python
Feature: Adds two integer features
Usefulness: This feature is needed for predicting the target.
Input samples: ’Number′
1 : [1, 0, 3],′ N umber′
2 : [0, 2, 1]
df[’Sum Number 1 and 2’] = df[’Number 1’] + df[’Number 2’]
“‘
““python
Feature: Removing the ’Name’ feature
Usefulness: This feature does not affect targeting
df.drop(columns=[’Name’], inplace=True)
“‘
[INIT PROMPT]:
{INIT_PROMPT}
< ACTOR LLM ANSWER >
{FIRST_LLM_ANSWER}
< LLM CRITIC ANSWER >
"""
