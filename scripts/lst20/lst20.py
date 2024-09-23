from argparse import ArgumentParser
from datasets import load_dataset
from prompt import instruction_prompt
from seqeval.metrics import classification_report
from tqdm import tqdm
from model_evaluation.models import get_model
import ast
import json


def ner(
    model,
    split="test",
    lst20_path="",
    limit=None,
) -> None:

    dataset = load_dataset("lst-nectec/lst20", split=split, data_dir=lst20_path)
    if limit == None:
        limit = len(dataset)
    ds = dataset.select(range(limit))

    results = []
    for item in tqdm(ds, total=len(ds)):
        labels = [
            dataset.features["ner_tags"].feature.int2str(idx)
            for idx in item["ner_tags"]
        ]
        try:
            response = model.inference(prompt=instruction_prompt.format(item["tokens"]))
            results.append(
                {
                    "tokens": item["tokens"],
                    "label": labels,
                    "pred": ast.literal_eval(response),
                }
            )
        except KeyError:
            pass

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    labels = []
    preds = []

    for ele in results:
        l = [item.replace("_", "-") for item in ele["label"]]
        p = [item[1].replace("_", "-") for item in ele["pred"]]
        if len(p) != len(l):
            n = min(len(p), len(l))
            p = p[:n]
            l = l[:n]
        labels.append(l)
        preds.append(p)
    report = classification_report(labels, preds)
    print(report)

    with open("report.txt", "w") as f:
        f.write(report)


def main() -> None:
    parser = ArgumentParser(description="Evaluate a model on the LST20 dataset for NER task.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use. Refer to model_evaluation/models.py for available models.")
    parser.add_argument("--lst20_path", type=str, required=True, help="Path to the LST20 dataset directory.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on (e.g., 'train', 'test', 'validation').")
    parser.add_argument("--limit", type=int, default=None, help="Number of samples to evaluate. If None, evaluate on the entire split.")
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for accessing the LLM model. Required if using a cloud-based LLM."
    )
    args = parser.parse_args()
    
    ner(get_model(args.model, args.api_key), args.lst20_path, args.split, args.limit)


if __name__ == "__main__":
    main()
    # from model_evaluation.models.openthaigpt_hf_7b_2023 import OpenThaiGPTHF7B2023

    # model = OpenThaiGPTHF7B2023(model_name="openthaigpt/openthaigpt-1.0.0-7b-chat")
    # ner(
    #     model,
    #     "test",
    #     "D:\Google Drive Mirroring\SuperAI\MT@LST-NECTEC\lst20\LST20_Corpus",
    #     10,
    # )
