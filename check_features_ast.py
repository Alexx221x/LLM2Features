import ast


LLM_PYTHON_ANSWER = "df['Title'] = df['Name'].str("

print(ast.parse(LLM_PYTHON_ANSWER))
