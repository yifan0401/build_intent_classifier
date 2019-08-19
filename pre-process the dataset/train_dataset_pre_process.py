import jsonlines
import pandas as pd



all_output = []
with jsonlines.open('train.jsonl') as f:
    for line in f:
        line = line['annotations']
        line_intent = line['intents']
        class_intent = line_intent['intent']
        intent_output = {'class':class_intent ,'annotations':{'intents':line_intent}}
        all_output.append(intent_output)

    df_intent = pd.DataFrame(all_output)

    df_intent = df_intent[['class', 'annotations']]

    df_utterance = pd.read_json('experiments/train.jsonl', lines=True)
    df_utterance = df_utterance[['prev_sys_utterance', 'prev_user_utterance', 'user_utterance', 'bot_name']]

    df_final = pd.concat([df_intent, df_utterance], axis=1)
    df_final.columns = ['class', 'annotations', 'prev_sys_utterance', 'prev_user_utterance', 'user_utterance','bot_name']


    out_final = df_final.to_json(orient='records', lines=True)

    with open('train_original_final.jsonl', 'w') as file:
        file.write(out_final)
        print('done!')




