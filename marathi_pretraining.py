#download marathi tweets, run this file and use the saved model for classification
from transformers import AutoTokenizer, AutoModelForMaskedLM, LineByLineTextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/marathi-tweets-bert")
model = AutoModelForMaskedLM.from_pretrained("snowood1/ConfliBERT-scr-uncased")

# Load the Marathi tweets dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./marathi-tweets.txt",  # Replace with the path to the Marathi tweets dataset file
    block_size=512,
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./ConfliBERT-scr-uncased-pretraining",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=5e-5,
    warmup_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=500,
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    use_wandb=False,
)

# Train the model
trainer.train()

# Save the pretrained model
trainer.save_model("./ConfliBERT-scr-uncased-pretraining/saved_model")