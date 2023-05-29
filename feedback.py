import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

base = os.getcwd()
model = tf.keras.models.load_model(base+"\\model\\feedback_rating.h5")
with open(base+'\\tokenizers\\feedback_rating_tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

def feedback_level(sentence):
    test_seq_pad = pad_sequences(tokenizer.texts_to_sequences([sentence]),maxlen=16, padding='post')
    predictions = model.predict(test_seq_pad, verbose=0)
    lvl = int(predictions[0][0]*5)
    print(f"Feedback Rating: {lvl}")

print("===========================================\n"*2, "Assume you were dining in a restaurant\n","===========================================\n")
while True:
    msg = input("What's your feedback: ")
    if msg == 'exit': break
    resp = feedback_level(msg)
    print(resp, "\n\n")