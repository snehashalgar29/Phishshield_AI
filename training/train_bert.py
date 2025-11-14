# Fine-tune BERT (requires internet + GPU ideally)
# Usage: python training/train_bert.py --dataset data/phish_dataset.csv --output_dir models/bert_phish --epochs 3
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/phish_dataset.csv')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='models/bert_phish')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    ds = load_dataset('csv', data_files=args.dataset)['train']
    ds = ds.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=256)
    tokenized = ds.map(preprocess, batched=True)
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    training_args = TrainingArguments(output_dir=args.output_dir, evaluation_strategy='epoch', num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size, load_best_model_at_end=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized['train'], eval_dataset=tokenized['test'], compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print('Saved BERT model to', args.output_dir)

if __name__ == '__main__':
    main()
