from openai import OpenAI
from pathlib import Path

from utils import cosine_similarity, hide_card_number_info, hide_cvv_info

client = OpenAI()

card_number_details_text = "My credit card number is 1234 5678 9012 3456."

card_number_embedding = client.embeddings.create(
  input=[card_number_details_text], 
  model="text-embedding-3-small"
).data[0].embedding

cvv_details_text = "My cvv is 123."

cvv_embedding = client.embeddings.create(
  input=[cvv_details_text], 
  model="text-embedding-3-small"
).data[0].embedding

combined_credit_card_cvv_text = "My credit card number is 1234 5678 9012 3456 and my cvv is 123."

combined_embedding = client.embeddings.create(
  input=[combined_credit_card_cvv_text], 
  model="text-embedding-3-small"
).data[0].embedding

log = Path('log.txt').read_text()

sentences = log.split('\n')
sentences = [sentence for sentence in sentences if len(sentence)>0]

embeddings = client.embeddings.create(
  input=sentences, model="text-embedding-3-small"
).data

for i, embedding_obj in enumerate(embeddings):
  embedding = embedding_obj.embedding

  card_number_similarity = cosine_similarity(card_number_embedding, embedding)
  cvv_similarity = cosine_similarity(cvv_embedding, embedding)
  combined_similarity = cosine_similarity(combined_embedding, embedding)

  if combined_similarity > 0.7:
    sentences[i] = hide_card_number_info(sentences[i])
    sentences[i] = hide_cvv_info(sentences[i])
  
  elif card_number_similarity > 0.7:
    sentences[i] = hide_card_number_info(sentences[i])
  
  elif cvv_similarity > 0.7:
    sentences[i] = hide_cvv_info(sentences[i])


new_log = '\n\n'.join(sentences)

f = open("new_log.txt", "w")
f.write(new_log)
f.close()