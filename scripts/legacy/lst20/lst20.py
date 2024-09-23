import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset
from seqeval.metrics import classification_report
from tqdm import tqdm

def tokenizand_align_labels(example, tokenizer, task, start_end_token, label_all_tokens=True):
        tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
        labels_aligned = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels_aligned.append(start_end_token)
            elif word_idx != previous_word_idx:
                labels_aligned.append(example[f"{task}_tags"][word_idx])
            else:
                labels_aligned.append(example[f"{task}_tags"][word_idx] if label_all_tokens else start_end_token)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = labels_aligned
        return tokenized_inputs
    
def evaluate_model(pretrained, lst20_path, cuda, split="test", task="ner", token_mapping=None, DEBUG=False):
    """
    Evaluate a Token Classification model on the LST20 dataset.

    This function loads a pre-trained model and tokenizer, prepares the dataset,
    performs token classification, and evaluates the model's performance using
    classification metrics. The dataset is expected to be in the LST20 format and
    contains tokens and corresponding labels for the specified task (e.g., named
    entity recognition).
    """

    # Load the model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(pretrained)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    model.to(device)

    # Load the dataset
    dataset = load_dataset("lst-nectec/lst20", split=split, data_dir=lst20_path)
    # dataset = dataset.select(range(100))

    # tokenize data and align new labels
    start_end_token = dataset.features[f"{task}_tags"].feature.str2int('0')
    dataset = dataset.map(lambda x: tokenizand_align_labels(x, tokenizer, task, start_end_token=start_end_token))

    # Prepare lists to store true and predicted labels
    grounds = []
    preds = []

    # Loop through the dataset
    for item in tqdm(dataset, total=len(dataset)):
        input_ids = torch.tensor([item["input_ids"]]).to(device)
        attention_mask = torch.tensor([item["attention_mask"]]).to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        # Get predicted labels
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        predicted_labels = [model.config.id2label[label_id.item()] for label_id in predictions]

        # mapping prediction token into LST20
        if token_mapping:
            predicted_labels_after_mapping = [token_mapping[x] if x in token_mapping else x for x in predicted_labels]
            preds.append(predicted_labels_after_mapping)
        else:
            preds.append(predicted_labels)

        # Convert true label IDs to labels
        true_labels = [dataset.features[f"{task}_tags"].feature.int2str(label_id) for label_id in item["labels"]]

        # seqeval prefer "_" to "-"
        true_labels = [label.replace("_", "-") for label in true_labels]
        
        grounds.append(true_labels)

        if DEBUG:
            print("Tokens:", item["tokens"])
            print(f"True Labels ({len(true_labels)}):", true_labels)
            print(f"Predicted Labels {len(predicted_labels)}:", predicted_labels)
            if token_mapping:
                print(f"Predicted Labels AF mapping{len(predicted_labels_after_mapping)}:", predicted_labels_after_mapping)
            print("---")

    # Evaluate the predictions using seqeval
    print("\nClassification Report:")
    print(classification_report(grounds, preds))


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model for token classification using seqeval.")
    parser.add_argument("--pretrained", type=str, required=True, help="The pretrained name or path (e.g., bert-base-cased).")
    parser.add_argument("--lst20_path", type=str, required=True, help="The Hugging Face dataset name (e.g., conll2003).")
    parser.add_argument("--split", type=str, default="test", help="The dataset split to evaluate on (default: test).")
    parser.add_argument("--cuda", default=False, action=argparse.BooleanOptionalAction, help="cuda flag")
    parser.add_argument("--task", type=str, default="ner", choices=["ner", "pos"], help="Task to perform")
    parser.add_argument("--token_mapping_path", type=str, default=None, help="Path to the token mapping file.")
    parser.add_argument("--debug", default=False, action=argparse.BooleanOptionalAction, help="Debug flag")    # parser.add_argument("--task", type=str, default="ner", choices=["ner", "pos"], help="Task to perform")
    args = parser.parse_args()

    # # Call the evaluation function
    token_mapping_dict = args.token_mapping_path
    if token_mapping_dict:
        path =  args.token_mapping_path
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                token_mapping_dict = {}
                for line in f.read().strip().split("\n"):
                    value = line.split()[0]
                    for key in line.split()[1:]:
                        token_mapping_dict[key] = value
        except Exception:
            raise(f"token_mapping: {path} is wrong format.")

    print(token_mapping_dict)
    
    evaluate_model(args.pretrained, args.lst20_path, args.cuda, args.split, args.task, token_mapping_dict,  args.debug)
    
        
    # evaluate_model(pretrained="pythainlp/thainer-corpus-v2-base-model",
    #                lst20_path="AIFORTHAI-LST20Corpus/LST20_Corpus",
    #                cuda=True,
    #                split="test",
    #                task="ner", token_mapping=token_mapping_dict, DEBUG=False)
    #example
    # python lst20.py --pretrained "KoichiYasuoka/bert-base-thai-upos" --lst20_path "AIFORTHAI-LST20Corpus/LST20_Corpus" --cuda --split "test" --task "pos"
    # python lst20.py --pretrained "pythainlp/thainer-corpus-v2-base-model" --lst20_path "AIFORTHAI-LST20Corpus/LST20_Corpus" --cuda --split "test" --task "ner" --token_mapping_path  "token_mapping.txt"

if __name__ == "__main__":
    main()
