import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

base = os.getcwd()
model = tf.keras.models.load_model(base+"\\model\\sarcasm_level.h5")
with open(base+'\\tokenizers\\sarcasm_tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

max_length=16

def test_level_of_sarcasm(sentence):
    test_seq_pad = pad_sequences(tokenizer.texts_to_sequences([sentence]),maxlen=max_length, padding='post')
    predictions = model.predict(test_seq_pad, verbose=0)
    lvl = int(predictions[0][0]*100)
    return f"Your thought is {lvl}% sarcastic"


print("===========================================\n"*2, "Judged according to bazzinga headline corpora\n", "===========================================\n")
while True:
    msg = input("What's in your mind?: ")
    if msg == 'exit': break
    resp = test_level_of_sarcasm(msg)
    print(resp, "\n\n")