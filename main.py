
from flask import Flask, request, render_template, url_for,jsonify
import os
app = Flask(__name__)

import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# Load the book text and process it
with open('the-ugly-duckling.txt', 'r', encoding='utf-8') as file:
    book_text = file.read().replace('\n', ' ')

# Split the book text into smaller chunks
max_length = 512  # Maximum sequence length for BERT models
book_chunks = [book_text[i:i+max_length] for i in range(0, len(book_text), max_length)]

num_epochs = 20

# Fine-tune the model on each book chunk
model.train()
for chunk in book_chunks:
    # Tokenize the book chunk
    inputs = tokenizer.encode_plus(chunk, return_tensors='pt', add_special_tokens=True)

    # Get the input IDs and attention mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Define the start and end positions for training the model
    start_positions = torch.tensor([0])
    end_positions = torch.tensor([len(input_ids[0]) - 1])

    # Prepare the training data
    train_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

    # Define the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=1e-5)
    total_steps = len(input_ids) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Fine-tune the model on the current chunk
    optimizer.zero_grad()
    outputs = model(**train_data)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    scheduler.step()


# Save the trained model
model.save_pretrained('trained_model')
tokenizer.save_pretrained('trained_model')


import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
model_sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')



import re

sentences = []

# Read the text file
with open('the-ugly-duckling.txt', 'r') as file:
    content = file.read()

# Remove unnecessary line breaks
content = re.sub(r'\n+', ' ', content)

# Split the content into sentences
sentences.extend(re.split(r'\.\s+', content))

# Remove empty sentences
sentences = [sentence for sentence in sentences if sentence]


def get_answer_from_book(question, context):
  # Tokenize the question and context
  inputs = tokenizer.encode_plus(question, context, return_tensors='pt', add_special_tokens=True)

  # Get the input IDs and attention mask
  input_ids = inputs['input_ids']
  attention_mask = inputs['attention_mask']

  # Use the model to predict the answer
  model.eval()
  with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      start_scores = outputs.start_logits
      end_scores = outputs.end_logits

  start_index = torch.argmax(start_scores)
  end_index = torch.argmax(end_scores)

  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index+1]))
  return( answer)


def main(question):
  context_made = ""

  top_sentences = get_most_similar_answers(question, sentences, model_sentence_transformer)

  for sentence, similarity in top_sentences:
      print(f'Sentence: {sentence}')
      context_made = context_made+sentence +". "
      print(f'Similarity: {similarity}')
      print('--------------------------')

  print("context made by top 5 similar sentences:" + context_made)
  print(get_answer_from_book(question,context_made))
  return(get_answer_from_book(question,context_made))


def get_most_similar_answers(question, answers, model):
    question_embedding = model.encode(question)
    answer_embeddings = [model.encode(answer) for answer in answers]

    cos_sim = [util.cos_sim(question_embedding, answer_embedding) for answer_embedding in answer_embeddings]

    answer_scores = list(zip(answers, cos_sim))
    answer_scores = sorted(answer_scores, key=lambda x: x[1], reverse=True)
    return answer_scores[:5]


@app.route('/')
def index():
    # Pass the URL to the template
    url = url_for('index', _external=True)
    return render_template('frontend.html', url=url)

@app.route('/index', methods=['POST'])
def process_string():
    data = request.get_json()
    input_string = data['input_string']
    result = main(input_string)

    response = {'result': result}
    return jsonify(result)


@app.route('/index', methods=['GET'])
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


