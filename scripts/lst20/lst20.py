import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics import classification_report

mapping = {
    'B_PERSON': 'B_PER',
    'I_PERSON': 'I_PER',
    'O': 'O',
    'B_ORGANIZATION': 'B_ORG',
    'B_LOCATION': 'B_LOC',
    'I_ORGANIZATION': 'I_ORG',
    'I_LOCATION': 'I_LOC',
    'B_DATE': 'B_DTM',
    'I_DATE': 'I_DTM',
    'B_TIME': 'B_DTM',  # Assuming TIME maps to DTM (DateTime)
    'I_TIME': 'I_DTM',  # Assuming TIME maps to DTM (DateTime)
    'B_MONEY': 'B_NUM',  # Assuming MONEY maps to NUM (Number)
    'I_MONEY': 'I_NUM',  # Assuming MONEY maps to NUM (Number)
    'B_FACILITY': 'B_LOC',  # Assuming FACILITY maps to LOC (Location)
    'I_FACILITY': 'I_LOC',  # Assuming FACILITY maps to LOC (Location)
    'B_URL': 'B_TRM',  # Assuming URL maps to TRM (Term)
    'I_URL': 'I_TRM',  # Assuming URL maps to TRM (Term)
    'B_PERCENT': 'B_NUM',  # Assuming PERCENT maps to NUM (Number)
    'I_PERCENT': 'I_NUM',  # Assuming PERCENT maps to NUM (Number)
    'B_LEN': 'B_NUM',  # Assuming LEN maps to NUM (Length as Number)
    'I_LEN': 'I_NUM',  # Assuming LEN maps to NUM (Length as Number)
    'B_AGO': 'B_DTM',  # Assuming AGO maps to DTM (DateTime)
    'I_AGO': 'I_DTM',  # Assuming AGO maps to DTM (DateTime)
    'B_LAW': 'B_TTL',  # Assuming LAW maps to TTL (Title)
    'I_LAW': 'I_TTL',  # Assuming LAW maps to TTL (Title)
    'B_PHONE': 'B_NUM',  # Assuming PHONE maps to NUM (Phone as Number)
    'I_PHONE': 'I_NUM',  # Assuming PHONE maps to NUM (Phone as Number)
    'B_EMAIL': 'B_TRM',  # Assuming EMAIL maps to TRM (Term)
    'I_EMAIL': 'I_TRM',  # Assuming EMAIL maps to TRM (Term)
    'B_ZIP': 'B_NUM',  # Assuming ZIP maps to NUM (Zip as Number)
    'B_TEMPERATURE': 'B_NUM',  # Assuming TEMPERATURE maps to NUM (Temperature as Number)
    'I_TEMPERATURE': 'I_NUM',  # Assuming TEMPERATURE maps to NUM (Temperature as Number)
    'B_DTAE': 'B_DTM',  # Assuming DTAE (likely DATE typo) maps to DTM
    'I_DTAE': 'I_DTM',  # Assuming DTAE (likely DATE typo) maps to DTM
    'B_DATA': 'B_TRM',  # Assuming DATA maps to TRM (Term)
    'I_DATA': 'I_TRM'   # Assuming DATA maps to TRM (Term)
}

def evaluate_model(pretrained, lst20_path, cuda, split="test", task="ner"):
    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    model.to(device)

    # Load the dataset
    dataset = load_dataset("lst-nectec/lst20", split=split, data_dir=lst20_path)
    dataset = dataset.select(range(10))
    labels = dataset.features[f"{task}_tags"].feature.names

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        label_all_tokens = True
        labels_aligned = []

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels_aligned.append(-100)
            elif word_idx != previous_word_idx:
                labels_aligned.append(example[f"{task}_tags"][word_idx])
            else:
                labels_aligned.append(example[f"{task}_tags"][word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = labels_aligned
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels)
    
    # Prepare lists to store true and predicted labels
    true = []
    pred = []

    # Loop through the dataset
    for item in dataset:
        input_ids = torch.tensor([item['input_ids']]).to(device)
        attention_mask = torch.tensor([item['attention_mask']]).to(device)
        labels = torch.tensor([item['labels']]).to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Get predicted labels
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        predicted_labels = [model.config.id2label[label_id.item()].replace("-", "_") for label_id in predictions]
        # print(model.config.id2label)
        # print(dataset.features["ner_tags"].feature.int2str)
        # print(predicted_labels)
        predicted_labels = [mapping.get(label) for label in predicted_labels]
        pred.append(predicted_labels)
        
        
        # Convert true label IDs to labels
        labels = item[f"labels"][1:-2]
        labels.insert(0, 0)
        labels.append(0)
        true_labels = [dataset.features["ner_tags"].feature.int2str(label_id) for label_id in labels]
        

        # Adjust the lengths of true and predicted labels
        if len(true_labels) < len(predicted_labels):
            true_labels += ["O"] * (len(predicted_labels) - len(true_labels))
        elif len(true_labels) > len(predicted_labels):
            true_labels = true_labels[:len(predicted_labels)]

        true.append(true_labels)

        # print("Tokens:", item["tokens"])
        # print("True Labels:", true_labels)
        # print("Predicted Labels:", predicted_labels)
        # print("---")
    
    # Evaluate the predictions using seqeval
    print("\nClassification Report:")
    print(classification_report(true, pred))

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model for token classification using seqeval.")
    parser.add_argument("--pretrained", type=str, required=True, help="The pretrained name or path (e.g., bert-base-cased).")
    parser.add_argument("--lst20_path", type=str, required=True, help="The Hugging Face dataset name (e.g., conll2003).")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate on (default: test).")
    parser.add_argument('--cuda', default=False, action=argparse.BooleanOptionalAction, help='cuda flag')
    parser.add_argument('--task', type=str, default="ner", choices=["ner", "pos"], help='Task to perform')
    args = parser.parse_args()

    # Call the evaluation function
    evaluate_model(args.pretrained, args.lst20_path, args.cuda, args.split, args.task)

if __name__ == "__main__":
    main()
