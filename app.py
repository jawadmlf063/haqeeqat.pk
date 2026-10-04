from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

app = Flask(_name_)

# Path to your fine-tuned model and tokenizer
MODEL_PATH = 'path/to/your/fine-tuned/model'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['news']
    
    inputs = tokenizer(data, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = F.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1)
    
    result = 'Fake News' if prediction.item() == 0 else 'Real News'
    return jsonify({'result': result})

if _name_ == '_main_':
    app.run(debug=True)
