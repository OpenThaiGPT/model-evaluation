from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_metric
from tqdm import tqdm
import argparse
import json
import os


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
    out_path,
):
    """
    Evaluate a sequence-to-sequence model on a given dataset and compute ROUGE and BLEU scores.

    This function loads a pre-trained sequence-to-sequence model and tokenizer, processes the dataset,
    generates summaries for each example in the dataset, and evaluates the generated summaries using
    ROUGE and BLEU metrics. Results are saved to specified output files.

    Args:
        pretrained (str): The identifier or path for the pre-trained model and tokenizer.
        dataset (str): The name or path of the dataset to evaluate.
        split (str): The dataset split to use (e.g., "train", "test").
        out_path (str): The directory path where the evaluation results will be saved.
        progress_bar (bool, optional): Whether to display a progress bar during summary generation.
                                        Default is True.

    Returns:
        None: This function prints out ROUGE and BLEU scores and saves the results to JSON files.

    Example:
        evaluate_model(
            pretrained="facebook/bart-large-cnn",
            dataset="cnn_dailymail",
            split="test",
            out_path="/path/to/output",
            progress_bar=True
        )
    """

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained)
    dataset = load_dataset(dataset, split=split)
    # dataset = dataset.select(range(3))
    # load rouge & blue
    rouge = load_metric("rouge", trust_remote_code=True)
    bleu = load_metric("bleu", trust_remote_code=True)

    # Calculate ROUGE and BLEU scores
    references = []
    hypotheses = []

    # Loop for predictions
    with tqdm(total=len(dataset), desc="Generating Summaries") as pbar:
        for example in dataset:
            reference = example["summary"]
            hypothesis = generate_summary(example["body"], model, tokenizer)
            references.append(reference)
            hypotheses.append(hypothesis)
            pbar.update(1)

    # C ROUGE scores
    rouge_result = rouge.compute(predictions=hypotheses, references=references, use_aggregator=True)

    # Compute BLEU scores (Note: BLEU expects tokenized input)
    bleu_result = bleu.compute(
        predictions=[hyp.split() for hyp in hypotheses],
        references=[[ref.split()] for ref in references],
    )

    # report results
    print("ROUGE Scores:", rouge_result)
    print("BLEU Score:", bleu_result)

    # save path
    with open(os.path.join(out_path, "rouge_result.json"), "w") as f:
        json.dump(rouge_result, f, indent=4)

    with open(os.path.join(out_path, "bleu_result.json"), "w") as f:
        json.dump(bleu_result, f, indent=4)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Evaluate a model for ThaiSum by Rouge, Bleu score")
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
    parser.add_argument(
        "--out_path",
        type=str,
        default=".",
        help="save_path for results.",
    )

    args = parser.parse_args()

    # Call the evaluation function
    evaluate_model(args.pretrained, args.dataset, args.split, args.out_path)


if __name__ == "__main__":
    main()
