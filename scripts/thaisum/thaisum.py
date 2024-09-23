from argparse import ArgumentParser
from datasets import load_dataset
from prompt import instruction_prompt
from tqdm import tqdm
from model_evaluation.models import get_model
import evaluate
from attacut import tokenize
import json

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


def thaisum(
    model,
    thaisum_path="nakhun/thaisum",
    split="test",
    limit=None,
) -> None:

    dataset = load_dataset(thaisum_path, split=split)
    if limit == None:
        limit = len(dataset)
    dataset = dataset.select(range(limit))

    results = []
    for item in tqdm(dataset, total=len(dataset)):
        try:
            response = model.inference(
                prompt=instruction_prompt.format(item["title"], item["body"])
            )
            results.append(
                {
                    "title": item["title"],
                    "body": item["body"],
                    "reference": item["summary"],
                    "prediction": response,
                    "tags": item["tags"],
                    "url": item["url"],
                }
            )
        except KeyError:
            pass

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    preds = [ele["prediction"] for ele in results]
    refs = [ele["reference"] for ele in results]
    preds = [" ".join(tokenize(p)) for p in preds]
    refs = [" ".join(tokenize(r)) for r in refs]

    # Compute ROUGE scores
    rouge_result = rouge.compute(
        predictions=preds, references=refs, use_aggregator=True
    )

    # Compute BLEU scores
    bleu_result = bleu.compute(
        predictions=preds,
        references=refs,
    )

    # report results
    print("ROUGE Scores:", rouge_result)
    print("BLEU Score:", bleu_result)

    # save path
    with open("rouge_result.json", "w") as f:
        json.dump(rouge_result, f, indent=4)

    with open("bleu_result.json", "w") as f:
        json.dump(bleu_result, f, indent=4)


def main() -> None:
    parser = ArgumentParser(description="ThaiSum inference")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model file",
    )

    parser.add_argument(
        "--thaisum_path",
        type=str,
        default="nakhun/thaisum",
        help="Path to the ThaiSum dataset",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Split of the dataset to use",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to process",
    )

    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for accessing the LLM model. Required if using a cloud-based LLM.",
    )

    args = parser.parse_args()
    thaisum(
        model=get_model(args.model, args.api_key),
        thaisum_path=args.thaisum_path,
        split=args.split,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
