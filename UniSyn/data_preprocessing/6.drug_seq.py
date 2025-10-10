import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
import pickle


data = pd.read_csv("./Drug_smiles.csv")

tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


embeddings = []
for _, row in data.iterrows():
    smiles = row['SMILES']
    drug_id = row['Drug']

  
    encoded_input = tokenizer(smiles, max_length=512, truncation=True, return_tensors="pt")

   
    outputs = model(**encoded_input)
    last_hidden_states = outputs.last_hidden_state

  
    embedding = last_hidden_states[:, 0, :].squeeze(0).detach().numpy()
 
    embeddings.append([drug_id] + embedding.tolist())

embedding_dim = len(embeddings[0]) - 1  
columns = ["Drug_ID"] + [f"dim_{i}" for i in range(embedding_dim)]


embedding_df = pd.DataFrame(embeddings, columns=columns)

embedding_df.to_csv("./drug_sequence_em.csv",index=False)


print("Saved drug embeddings with drug IDs.")