from transformers import (
    AutoTokenizer, 
    AutoModel, # If we only need the embeddings
    AutoModelForMaskedLM # If we want to work with the masked LM embeddings (additional logits output)
)
import torch
import pandas as pd
import random 

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu2'
    print(f"Using device: {device}")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    # Load ESM2 8M model and tokenizer
    model_checkpoint = "facebook/esm2_t6_8M_UR50D"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download=True)

    # TODO: for later usage with LeoMed, save your model (and tokenizer) and scp this directory to the server
    # Then set force_download=False if you specify a local path to load from 
    # path_to_save = ...
    # model.save_pretrained(path_to_save) 
    
 
    # Load sample DMS data
    df = pd.read_csv('data/gfp_ground_truth.csv')

    # Extract sample from df and mask random 15%
    seq = df['sequence'][0]
    mask_prob = 0.15
    masked_chars = []
    for aa in seq:
        if random.random() < mask_prob:
            masked_chars.append(tokenizer.mask_token)
        else:
            masked_chars.append(aa)
    masked_seq = "".join(masked_chars)

    # Tokenize and input to ESM2
    inputs = tokenizer(masked_seq, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Decode predictions by taking argmax
    pred_ids = logits.argmax(dim=-1)  # (1, seq_len)
    pred_seq = tokenizer.decode(
        pred_ids[0],
        skip_special_tokens=True
    ).replace(" ", "")

    print("\nGround truth sequence:")
    print(seq)
    print("\nMasked input sequence:")
    print(masked_seq)
    print("\nPredicted sequence:")
    print(pred_seq)
