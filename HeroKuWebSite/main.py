##from flask_ngrok import run_with_ngrok
from flask import Flask,request,jsonify, render_template
from datetime import datetime
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import bert 
import os
import numpy
app=Flask(__name__)
##run_with_ngrok(app)

BertTokenizer=bert.bert_tokenization.FullTokenizer
bert_layer=hub.KerasLayer('https://tfhub.dev/tensorflow/bert_zh_L-12_H-768_A-12/2',trainable=False)
vocabulary_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
tokenizer=BertTokenizer(vocabulary_file,to_lower_case)
max_len=500
BATCH_SIZE=100
new_model=tf.keras.models.load_model('BERT_20210830_Psychotic_depression_model')

@app.route('/',methods=['POST','GET'])
def index():
  title="輸入病史資料(建議100字以上)"+str(datetime.today())
  myText=request.form.get('myHistory')
  if myText:
    myText_bert=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(str(myText)))
    test_sorted_text_labels=[(myText_bert[:max_len])]
    test_processed=tf.data.Dataset.from_generator(lambda:test_sorted_text_labels,output_types=(tf.int32))
    test_batched=test_processed.padded_batch(BATCH_SIZE,padded_shapes=((max_len,)))
    result=new_model.predict(test_batched)
    myPredict=round(result[0][0],5)
  else:
    myPredict=0
  ##myPredict='50%'
  return render_template("index.html",title=title,myHx=myText,myPredict=myPredict)

app.run()