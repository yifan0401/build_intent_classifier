import pandas as pd
from sklearn.model_selection import train_test_split


df_final = pd.read_json("train_original_final.jsonl", lines=True)

# Split train set into new train set and test set

X, y = df_final.loc[:, ['class', 'annotations', 'prev_sys_utterance', 'prev_user_utterance', 'user_utterance','bot_name']], \
       df_final.loc[:, ['annotations']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.00278, stratify=y)

# Save files into new train set and test set

out_final = df_final.to_json(orient='records', lines=True)

c = X_train.to_json(orient='records', lines=True)
d = X_test.to_json(orient='records', lines=True)
with open('train_new.jsonl', 'w') as file:
    file.write(c)
    print('done!train!')
with open('test_new.jsonl', 'w') as file:
    file.write(d)
    print('done!test!')

print("done!")