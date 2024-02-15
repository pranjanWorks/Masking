import numpy as np
import re

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hide_card_number_info(text):
  card_number_pattern = r'\d{4}[\s\-]*?\d{4}[\s\-]*?\d{4}[\s\-]*?\d{4}'
  safe_text = re.sub(card_number_pattern, "xxxx-xxxx-xxxx-xxxx", text)
  return safe_text

def hide_cvv_info(text):
  cvv_pattern = r'\d{3}'
  safe_text = re.sub(cvv_pattern, "xxx", text)
  return safe_text