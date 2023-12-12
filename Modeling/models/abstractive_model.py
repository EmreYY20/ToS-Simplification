import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, tokenizer, df, max_length=512, max_target_length=128):
        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for _, row in df.iterrows():
            inputs = tokenizer.encode_plus(row['plain_text'], max_length=max_length, return_tensors='pt', truncation=True)
            labels = tokenizer.encode_plus(row['summary'], max_length=max_target_length, return_tensors='pt', truncation=True)

            self.input_ids.append(inputs.input_ids.flatten())
            self.attention_mask.append(inputs.attention_mask.flatten())
            self.labels.append(labels.input_ids.flatten())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 
                'attention_mask': self.attention_mask[idx], 
                'labels': self.labels[idx]}

try:
    # Load the fine-tuning dataset
    df = pd.read_json('Modeling\models\dataset\dataset.json', lines=True)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    # Load BART pre-trained model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Create datasets
    train_dataset = CustomDataset(tokenizer, train_df)
    val_dataset = CustomDataset(tokenizer, val_df)

    # Define training configurations
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=100,
        save_steps=500,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define the Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Start training using the Trainer instance
    trainer.train()

except Exception as e:
    print(f"An error occurred: {e}")