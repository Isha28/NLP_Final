# from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
# import torch

# # Download and load the "cardiffnlp/twitter-roberta-base" model
# model_name = "cardiffnlp/twitter-roberta-base"
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaForMaskedLM.from_pretrained(model_name)

# # Download and load the "snowood1/ConfliBERT-scr-uncased" model
# conflibert_model_name = "snowood1/ConfliBERT-scr-uncased"
# conflibert_model = RobertaForMaskedLM.from_pretrained(conflibert_model_name)

# # Download the dataset you want to use for pretraining your model
# # Preprocess the dataset, tokenize the text and apply the mask for the masked language modeling task
# # Here, we will use the Hugging Face Datasets library to download and preprocess the dataset
# from datasets import load_dataset
# dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
# text = dataset['train']['text']
# inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
# inputs = inputs['input_ids']

# # Fine-tune the "cardiffnlp/twitter-roberta-base" model on your preprocessed dataset using the masked language modeling objective
# # During fine-tuning, you can either use the same hyperparameters as the "cardiffnlp/twitter-roberta-base" model or experiment with different ones to optimize the performance of your model
# optimizer = AdamW(model.parameters(), lr=1e-5)
# model.train()
# for epoch in range(1): #3
#     for batch in inputs.split(256):
#         optimizer.zero_grad()
#         outputs = model(batch, labels=batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Once you have fine-tuned the "cardiffnlp/twitter-roberta-base" model on your preprocessed dataset, you can save the weights of the model to disk
# model.save_pretrained("finetuned_cardiffnlp")

# print ("DONE!!!!")

# # Load the saved fine-tuned model weights
# model = RobertaForMaskedLM.from_pretrained("cardiffnlp/twitter-roberta-base")

# # Replace the weights of the "snowood1/ConfliBERT-scr-uncased" model's masked language modeling head with the weights you saved from the fine-tuned "cardiffnlp/twitter-roberta-base" model
# conflibert_model.lm_head.load_state_dict(model.lm_head.state_dict())

# # You can now use the "snowood1/ConfliBERT-scr-uncased" model with the fine-tuned masked language modeling head to generate predictions on your downstream task
# conflibert_model.save_pretrained("finetuned_conflibert")

# WORKS!!!
# import torch
# from transformers import RobertaForMaskedLM

# conflibert_model_name = "snowood1/ConfliBERT-scr-uncased"
# conflibert_model = RobertaForMaskedLM.from_pretrained(conflibert_model_name)
# model = RobertaForMaskedLM.from_pretrained("cardiffnlp/twitter-roberta-base")

# # Create a new linear layer with the same size as the lm_head layer of the "cardiffnlp/twitter-roberta-base" model
# new_lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size)

# # Copy the weights of the lm_head layer of the "cardiffnlp/twitter-roberta-base" model to the new linear layer
# new_lm_head.weight = model.lm_head.decoder.weight
# new_lm_head.bias = model.lm_head.decoder.bias

# # Replace the lm_head layer of the "snowood1/ConfliBERT-scr-uncased" model with the new linear layer
# conflibert_model.lm_head = new_lm_head

# print ("DONE!!!!")


# from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
# from torch.utils.data import Dataset, DataLoader
# import torch

# # Download and load the "cardiffnlp/twitter-roberta-base" model
# model_name = "cardiffnlp/twitter-roberta-base"
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaForMaskedLM.from_pretrained(model_name)

# # Download and load the "snowood1/ConfliBERT-scr-uncased" model
# conflibert_model_name = "snowood1/ConfliBERT-scr-uncased"
# conflibert_model = RobertaForMaskedLM.from_pretrained(conflibert_model_name)

# # Define a custom dataset class that loads input data from disk
# class WikiTextDataset(Dataset):
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.lines = []
#         with open(self.file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 self.lines.append(line.strip())

#     def __getitem__(self, index):
#         line = self.lines[index]
#         input_ids = tokenizer.encode(line, padding='max_length', truncation=True, max_length=128)
#         return input_ids

#     def __len__(self):
#         return len(self.lines)

# # Load the dataset using a data loader
# dataset = WikiTextDataset('wikitext-2-raw/wiki.train.raw')
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # Fine-tune the "cardiffnlp/twitter-roberta-base" model on your preprocessed dataset using the masked language modeling objective
# # During fine-tuning, you can either use the same hyperparameters as the "cardiffnlp/twitter-roberta-base" model or experiment with different ones to optimize the performance of your model
# optimizer = AdamW(model.parameters(), lr=1e-5)
# model.train()
# for epoch in range(1): #3
#     for batch in dataloader:
#         optimizer.zero_grad()
#         batch = torch.stack(batch).to(torch.int64)
#         outputs = model(batch, labels=batch)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()

# # Once you have fine-tuned the "cardiffnlp/twitter-roberta-base" model on your preprocessed dataset, you can save the weights of the model to disk
# model.save_pretrained("finetuned_cardiffnlp")

# print ("DONE!!!!")


#fails because of GPU

# import torch
# from transformers import RobertaTokenizer, RobertaForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# # Define your 5 training sentences
# sentences = ["I love to eat pizza",
#              "The dog ran after the ball",
#              "She is studying for her exam",
#              "He enjoys playing video games",
#              "The sun is shining brightly today"]

# # Save the sentences to a text file
# with open('training_data.txt', 'w') as f:
#     for sentence in sentences:
#         f.write(sentence + '\n')

# print ("PRINT: Created text file for training")

# # Tokenize the sentences
# tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base')

# # Create a LineByLineTextDataset
# dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path='training_data.txt', block_size=128)

# # Mask some tokens in the input
# mask_token_index = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
# inputs = dataset[0]['input_ids']
# inputs[2] = mask_token_index

# # Create a DataCollatorForLanguageModeling
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# print ("PRINT: Created data collator")

# # Load the pre-trained model
# model = RobertaForMaskedLM.from_pretrained('snowood1/ConfliBERT-scr-uncased')

# print ("PRINT: Created model")

# # Define TrainingArguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     overwrite_output_dir=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=32,
#     save_steps=10_000,
#     save_total_limit=2
# )

# # Create a Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=data_collator
# )

# print ("PRINT: Created trainer")

# # Start training
# trainer.train()

# print ("PRINT: Completed training")

import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("snowood1/ConfliBERT-scr-uncased")
model = AutoModelForMaskedLM.from_pretrained("snowood1/ConfliBERT-scr-uncased")

# Load the data
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize the data
encoded = tokenizer(text, padding=True, truncation=True, return_tensors='tf')

# Mask some tokens
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']

# Mask 15% of the tokens
mask_prob = 0.15
mask_indices = np.random.binomial(1, mask_prob, size=input_ids.shape) == 1
mask_indices &= attention_mask  # ignore the padded tokens

# Replace the selected tokens with the [MASK] token
replace_prob = 0.8  # probability of replacing the masked token with [MASK]
keep_prob = 1 - replace_prob  # probability of keeping the original token
replace_indices = np.random.binomial(1, replace_prob, size=input_ids.shape) == 1
keep_indices = np.logical_not(replace_indices)

input_ids[mask_indices & replace_indices] = tokenizer.mask_token_id
input_ids[mask_indices & keep_indices] = input_ids[mask_indices & keep_indices]

# Add the special tokens
inputs = tokenizer.prepare_for_model(input_ids, attention_mask=attention_mask)

# Define the loss function and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Define the training loop
batch_size = 8
num_epochs = 1
num_train_steps = (len(text) // batch_size) * num_epochs

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    epoch_loss = 0

    for i in range(0, len(text), batch_size):
        # Prepare the batch
        batch = {k: v[i:i+batch_size] for k, v in inputs.items()}

        # Compute the loss
        with tf.GradientTape() as tape:
            logits = model(batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])[0]
            loss = loss_fn(batch['input_ids'][:, 1:], logits[:, :-1])

        # Compute the gradients
        grads = tape.gradient(loss, model.trainable_weights)

        # Update the model parameters
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch_loss += loss

        if (i // batch_size) % 100 == 0:
            print(f"Step {i // batch_size}/{num_train_steps // batch_size}, Loss: {loss.numpy()}")

    print(f"Epoch Loss: {epoch_loss.numpy()}")

# Save the trained model
model.save_pretrained("confliclm")
tokenizer.save_pretrained("confliclm")
print ("DONE!!!!")

