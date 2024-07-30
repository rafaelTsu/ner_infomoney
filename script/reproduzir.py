import json
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Carregar os dados exportados do Label Studio
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Preparar os dados para o formato Hugging Face
def prepare_data(data):
    tokens = []
    labels = []

    for item in data:
        text = item['data']['text']
        annotations = item.get('annotations', [{}])[0].get('result', [])
        
        token_labels = ['O'] * len(text.split())  # 'O' significa fora de uma entidade

        for ann in annotations:
            start = ann['value']['start']
            end = ann['value']['end']
            label = ann['value']['labels'][0]

            start_idx = 0
            for i, token in enumerate(text.split()):
                token_start = text.find(token, start_idx)
                token_end = token_start + len(token)
                start_idx = token_end

                if start <= token_start < end or start < token_end <= end:
                    token_labels[i] = f'B-{label}' if token_labels[i] == 'O' else f'I-{label}'
        
        tokens.append(text.split())
        labels.append(token_labels)

    return Dataset.from_dict({"tokens": tokens, "labels": labels})

# Função principal para treinar o modelo
def train_model(train_file_path, model_checkpoint, output_dir, label_to_id_mapping):
    data = load_data(train_file_path)
    dataset = prepare_data(data)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_to_id_mapping))

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id_mapping.get(label[word_idx], 0))  # Usar 0 para 'O'
                else:
                    label_ids.append(label_to_id_mapping.get(label[word_idx], 0) if label[word_idx].startswith('I-') else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Divida o conjunto de dados em treinamento e avaliação
    dataset_split = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[id_to_label_mapping[id] for id in label if id != -100] for label in labels]
        true_predictions = [[id_to_label_mapping[pred] for (pred, id) in zip(prediction, label) if id != -100]
                            for prediction, label in zip(predictions, labels)]

        true_labels_flat = [label for sublist in true_labels for label in sublist]
        true_predictions_flat = [pred for sublist in true_predictions for pred in sublist]

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels_flat, true_predictions_flat, average='weighted', zero_division=0)
        accuracy = accuracy_score(true_labels_flat, true_predictions_flat)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    # Caminho para o arquivo exportado do Label Studio
    train_file_path = "dataset/exported.json"
    # Checkpoint do modelo base
    model_checkpoint = "distilbert-base-cased"
    # Diretório de saída para salvar o modelo treinado
    output_dir = "script/output"

    # Mapeamento de rótulos para IDs
    label_to_id_mapping = {
        "O": 0,
        "empresa": 1,
        "empresario": 2,
        "politico": 3,
        "outras_pessoas": 4,
        "valor_financeiro": 5,
        "cidade": 6,
        "estado": 7,
        "pais": 8,
        "organizacao": 9,
        "banco": 10
    }

    # Inverter o mapeamento para avaliação
    id_to_label_mapping = {v: k for k, v in label_to_id_mapping.items()}

    # Treinar o modelo
    train_model(train_file_path, model_checkpoint, output_dir, label_to_id_mapping)