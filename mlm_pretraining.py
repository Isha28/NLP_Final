# from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW
# # import torch

# # # Download and load the "cardiffnlp/twitter-roberta-base" model
# # model_name = "cardiffnlp/twitter-roberta-base"
# # tokenizer = RobertaTokenizer.from_pretrained(model_name)
# # model = RobertaForMaskedLM.from_pretrained(model_name)

# # Download and load the "snowood1/ConfliBERT-scr-uncased" model
# conflibert_model_name = "snowood1/ConfliBERT-scr-uncased"
# conflibert_model = RobertaForMaskedLM.from_pretrained(conflibert_model_name)

# # # Download the dataset you want to use for pretraining your model
# # # Preprocess the dataset, tokenize the text and apply the mask for the masked language modeling task
# # # Here, we will use the Hugging Face Datasets library to download and preprocess the dataset
# # # from datasets import load_dataset
# # dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
# # text = dataset['train']['text']
# # inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
# # inputs = inputs['input_ids']

# # # Fine-tune the "cardiffnlp/twitter-roberta-base" model on your preprocessed dataset using the masked language modeling objective
# # # During fine-tuning, you can either use the same hyperparameters as the "cardiffnlp/twitter-roberta-base" model or experiment with different ones to optimize the performance of your model
# # optimizer = AdamW(model.parameters(), lr=1e-5)
# # model.train()
# # for epoch in range(1): #3
# #     for batch in inputs.split(256):
# #         optimizer.zero_grad()
# #         outputs = model(batch, labels=batch)
# #         loss = outputs.loss
# #         loss.backward()
# #         optimizer.step()

# # # Once you have fine-tuned the "cardiffnlp/twitter-roberta-base" model on your preprocessed dataset, you can save the weights of the model to disk
# # model.save_pretrained("finetuned_cardiffnlp")

# # Load the saved fine-tuned model weights
# model = RobertaForMaskedLM.from_pretrained("cardiffnlp/twitter-roberta-base")

# # Replace the weights of the "snowood1/ConfliBERT-scr-uncased" model's masked language modeling head with the weights you saved from the fine-tuned "cardiffnlp/twitter-roberta-base" model
# conflibert_model.lm_head.load_state_dict(model.lm_head.state_dict())

# # You can now use the "snowood1/ConfliBERT-scr-uncased" model with the fine-tuned masked language modeling head to generate predictions on your downstream task
# conflibert_model.save_pretrained("finetuned_conflibert")

import torch
from transformers import RobertaForMaskedLM

conflibert_model_name = "snowood1/ConfliBERT-scr-uncased"
conflibert_model = RobertaForMaskedLM.from_pretrained(conflibert_model_name)
model = RobertaForMaskedLM.from_pretrained("cardiffnlp/twitter-roberta-base")

# Create a new linear layer with the same size as the lm_head layer of the "cardiffnlp/twitter-roberta-base" model
new_lm_head = torch.nn.Linear(model.config.hidden_size, model.config.vocab_size)

# Copy the weights of the lm_head layer of the "cardiffnlp/twitter-roberta-base" model to the new linear layer
new_lm_head.weight = model.lm_head.decoder.weight
new_lm_head.bias = model.lm_head.decoder.bias

# Replace the lm_head layer of the "snowood1/ConfliBERT-scr-uncased" model with the new linear layer
conflibert_model.lm_head = new_lm_head

print ("DONE!!!!")
