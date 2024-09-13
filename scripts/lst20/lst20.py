import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset
from seqeval.metrics import classification_report

def evaluate_model(pretrained, lst20_path, cuda, split="test"):
    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() and cuda == True else "cpu")

    # Initialize the pipeline with token-classification
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)

    # Load the dataset
    dataset = load_dataset("lst-nectec/lst20", split=split, trust_remote_code=True, data_dir=lst20_path)
    
    # Prepare lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Loop through the dataset
    for item in dataset:
        
        text = item['tokens']
        true_label = item['ner_tags']  # Adjust this based on the dataset label field
        # Convert true label indices to label strings using dataset features
        true_label_str = [dataset.features['ner_tags'].feature.int2str(label) for label in true_label]

        text_str = " ".join(text)
        token_results = nlp(text_str)
        
        # Initialize predicted labels as 'O' (outside)
        labels = ['O'] * len(text)

        # Map the predictions back to the token level
        for result in token_results:
            start = result['start']
            stop = result['end']
            label = result['entity_group']
            labels[start:stop] = [label]
        # Append results
        predicted_labels.append(labels)
        true_labels.append(true_label_str)

    # Evaluate the predictions using seqeval
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Evaluate a model for token classification using seqeval.")
    parser.add_argument("--pretrained", type=str, required=True, help="The pretrained name or path (e.g., bert-base-cased).")
    parser.add_argument("--lst20_path", type=str, required=True, help="The Hugging Face dataset name (e.g., conll2003).")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate on (default: test).")
    parser.add_argument('--cuda', default=False, action=argparse.BooleanOptionalAction, help='cuda flag') 

    args = parser.parse_args()

    # Call the evaluation function
    evaluate_model(args.pretrained, args.lst20_path, args.cuda, args.split)

if __name__ == "__main__":
    main()
