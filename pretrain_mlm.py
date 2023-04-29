from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import pandas as pd
import torch
import os
os.environ["WANDB_DISABLED"] = "true"

def pretrain_conflibert(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)

    df = pd.read_csv('new_tweets_mlm.tsv', sep='\t')
    text = df.values.tolist()
    print("PRINT : Input Texts", text[:5])
    
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    class MeditationsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __getitem__(self, idx):
            return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        def __len__(self):
            return len(self.encodings.input_ids)

    dataset = MeditationsDataset(inputs)

    args = TrainingArguments(
        output_dir=model_path+'mlm_pretrain',
        per_device_train_batch_size=16,
        num_train_epochs=1, #change
        evaluation_strategy='epoch'
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    print ("PRINT: Completed training")
    model.save_pretrained(model_path+'mlm_pretrain')

def main():
    pretrain_conflibert('snowood1/ConfliBERT-scr-cased')
    # pretrain_conflibert("snowood1/ConfliBERT-scr-uncased") #change
    # pretrain_conflibert("snowood1/ConfliBERT-cont-cased")
    # pretrain_conflibert("snowood1/ConfliBERT-cont-uncased")

if __name__ == "__main__":
    main()