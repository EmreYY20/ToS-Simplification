import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
import pandas as pd

# Load the fine-tuning dataset
df = pd.read_json('dataset\dataset.json', lines=True)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Load BART pre-trained model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Tokenize the input text and summaries
def tokenize_data(df):
    inputs = tokenizer(df['plain_text'].tolist(), padding=True, truncation=True, return_tensors='pt')
    labels = tokenizer(df['summary'].tolist(), padding=True, truncation=True, return_tensors='pt')
    return inputs, labels

train_inputs, train_labels = tokenize_data(train_df)
val_inputs, val_labels = tokenize_data(val_df)

# Define training configurations
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_steps=100,
    save_steps=500,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)

# Define the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=DataCollatorForSeq2Seq.from_tensors(train_inputs, train_labels),
    eval_dataset=DataCollatorForSeq2Seq.from_tensors(val_inputs, val_labels)
)

# Start training using the Trainer instance
trainer.train()



