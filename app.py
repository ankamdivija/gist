from flask import Flask, jsonify, render_template, request, redirect
import torch
import logging
from transformers import BartTokenizer, BartForConditionalGeneration, EncoderDecoderModel, T5ForConditionalGeneration, BertTokenizer, T5Tokenizer

app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)  # Set log level to INFO
handler = logging.FileHandler('app.log')  # Log to a file
app.logger.addHandler(handler)

models = [BartForConditionalGeneration.from_pretrained("facebook/bart-base"),
          T5ForConditionalGeneration.from_pretrained("google-t5/t5-base"),
           EncoderDecoderModel.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")]
models[2].load_state_dict(torch.load('./models/patrickvonplaten_bert2bert_cnn_daily_mail.pth', map_location=torch.device('cpu')))
models[0].load_state_dict(torch.load('./models/facebook_bart-base.pth', map_location=torch.device('cpu')))
models[1].load_state_dict(torch.load('./models/google-t5_t5-base.pth', map_location=torch.device('cpu')))

models[2].eval()
models[0].eval()
models[1].eval()

tokenizers = [BartTokenizer.from_pretrained("facebook/bart-base"), 
                T5Tokenizer.from_pretrained("google-t5/t5-base"),
              BertTokenizer.from_pretrained("patrickvonplaten/bert2bert_cnn_daily_mail")]

@app.route('/')
def hello() :
    return 'Hey!'

@app.route('/home')
def homepage() :
    return render_template("index.html")

@app.route('/knowmore')
def knowmore() :
    return render_template("knowmore.html")

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data['text']
    inputs1 = tokenizers[2](text, return_tensors="pt", truncation=True, padding="longest", max_length=600)
    summary_ids1 = models[2].generate(inputs1.input_ids, max_length=400, num_beams=2, length_penalty=1.0, early_stopping=True)
    summary1 = tokenizers[2].decode(summary_ids1[0], skip_special_tokens=True)
    
    inputs2 = tokenizers[0](text, return_tensors="pt", truncation=True, padding="longest", max_length=600)
    summary_ids2 = models[0].generate(inputs2.input_ids, max_length=400, num_beams=2, length_penalty=1.0, early_stopping=True) #, early_stopping=True
    summary2 = tokenizers[0].decode(summary_ids2[0], skip_special_tokens=True)
    
    inputs3 = tokenizers[1](text, return_tensors="pt", truncation=True, padding="longest", max_length=600)
    summary_ids3 = models[1].generate(inputs3.input_ids, max_length=400, num_beams=2, length_penalty=1.0, early_stopping=True) #, early_stopping=True
    summary3 = tokenizers[1].decode(summary_ids3[0], skip_special_tokens=True)
    return jsonify({'summary1': summary1, 'summary2': summary2, 'summary3': summary3})

@app.route('/predict')
def predict() :
    return render_template("predict.html")

@app.route('/analyse')
def analyse() :
    return render_template("analyse.html")

if __name__ == '__main__':
    app.run(debug=True)