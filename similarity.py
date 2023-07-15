import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Loading the CSV file into a DataFrame.
df = pd.read_csv('Precily_Text_Similarity.csv')

# Loading the Universal Sentence Encoder model.
encoder = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')

# Function to encode two text strings to vectors using the above encoder model.
def encode(text):
  """Embeds a text string using the Universal Sentence Encoder."""
  if isinstance(text, str) and text.strip() != '':
    return encoder([text]).numpy()
  else:
    return np.zeros((1, 512))  # Return a zero vector if the text is empty .

# Calculate the semantic similarity between the text strings in each row.
for i in range(len(df)):
  text1 = df.loc[i, 'text1']
  text2 = df.loc[i, 'text2']
  encoding1 = encode(text1)
  encoding2 = encode(text2)
  similarity = np.dot(encoding1, encoding2.T) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
  df.loc[i, 'similarity'] = similarity[0, 0]  # Extract the scalar value from the 1x1 matrix.

# Save the DataFrame to a new CSV file.
df.to_csv('Similar_results.csv')
