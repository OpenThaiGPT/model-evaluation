from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
from tqdm import tqdm
import argparse
import json
<<<<<<< HEAD
import os
=======

>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e

def generate_summary(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def evaluate_model(
    pretrained,
    dataset,
    split,
<<<<<<< HEAD
    out_path,
    progress_bar=True,
):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
    dataset = load_dataset(dataset, split=split)
    # dataset = dataset.select(range(3))
    #load rouge & blue
=======
    progress_bar=True,
):
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
    dataset = load_dataset(dataset, trust_remote_code=True, split=split)
    dataset = dataset.select(range(3))
>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e
    rouge = load_metric("rouge", trust_remote_code=True)
    bleu = load_metric("bleu", trust_remote_code=True)

    # Calculate ROUGE and BLEU scores
    references = []
    hypotheses = []

<<<<<<< HEAD
    # Loop for predictions
=======
>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e
    if progress_bar:
        with tqdm(total=len(dataset), desc="Generating Summaries") as pbar:
            for example in dataset:
                reference = example["summary"]
                hypothesis = generate_summary(example["body"], model, tokenizer)
                references.append(reference)
                hypotheses.append(hypothesis)
                pbar.update(1)
    else:
        for example in dataset:
            reference = example["summary"]
            hypothesis = generate_summary(example["body"], model, tokenizer)
            references.append(reference)
            hypotheses.append(hypothesis)

<<<<<<< HEAD
    # C ROUGE scores
=======
    # Compute ROUGE scores
>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e
    rouge_result = rouge.compute(predictions=hypotheses, references=references, use_aggregator=True)

    # Compute BLEU scores (Note: BLEU expects tokenized input)
    bleu_result = bleu.compute(
        predictions=[hyp.split() for hyp in hypotheses],
        references=[[ref.split()] for ref in references],
    )

<<<<<<< HEAD
    # report results
    print("ROUGE Scores:", rouge_result)
    print("BLEU Score:", bleu_result)

    # save path
    with open(os.path.join(out_path, "rouge_result.json"), "w") as f:
        json.dump(rouge_result, f, indent=4)

    with open(os.path.join(out_path, "bleu_result.json"), "w") as f:
=======
    print("ROUGE Scores:", rouge_result)
    print("BLEU Score:", bleu_result)

    with open("rouge_result.json", "w") as f:
        json.dump(rouge_result, f, indent=4)

    with open("bleu_result.json", "w") as f:
>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e
        json.dump(bleu_result, f, indent=4)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate a model for ThaiSum by Rouge, Bleu score"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="The model name or path (e.g., bert-base-cased).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The Hugging Face dataset name (e.g., conll2003).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The dataset split to evaluate on (default: test).",
    )
<<<<<<< HEAD
    parser.add_argument(
        "--out_path",
        type=str,
        default=".",
        help="save_path for results.",
    )
=======
>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e

    args = parser.parse_args()

    # Call the evaluation function
<<<<<<< HEAD
    evaluate_model(args.pretrained, args.dataset, args.split, args.out_path)
=======
    evaluate_model(args.pretrained, args.dataset, args.split)
>>>>>>> a27ab3c5fd70c3f1caf6cb4c19c2d1161697316e


if __name__ == "__main__":
    main()
