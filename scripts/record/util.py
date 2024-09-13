import datasets
import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.api.metrics import metric_max_over_ground_truths


def doc_to_text(doc):
    initial_text, *highlights = doc["passage_TH"].strip().split("\n@highlight\n")
    text = initial_text + "\n\n"
    for highlight in highlights:
        text += f"  - {highlight}.\n"
    return text


def format_answer(query, entity):
    return f"  - {query}".replace("@placeholder", entity)


def doc_to_target(doc):
    # We only output the first correct entity in a answers_TH
    return format_answer(query=doc["query_TH"], entity=doc["answers_TH"][0])


def doc_to_choice(doc):
    return [format_answer(query=doc["query_TH"], entity=ans) for ans in doc["entities_TH"]]


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        return {
            "passage_TH": doc["passage_TH"],
            "query_TH": doc["query_TH"],
            "entities_TH": sorted(list(set(doc["entities_TH"]))),
            "answers_TH": sorted(list(set(doc["answers_TH"]))),
        }

    return dataset.map(_process_doc)


def process_results(doc, results):
    # ReCoRD's evaluation is actually deceptively simple:
    # - Pick the maximum likelihood prediction entity
    # - Evaluate the accuracy and token F1 PER EXAMPLE
    # - Average over all examples
    max_idx = np.argmax(np.array([result[0] for result in results]))

    prediction = doc["entities_TH"][max_idx]
    gold_label_set = doc["answers_TH"]
    f1 = metric_max_over_ground_truths(
        squad_metrics.compute_f1, prediction, gold_label_set
    )
    em = metric_max_over_ground_truths(
        squad_metrics.compute_exact, prediction, gold_label_set
    )

    return {
        "f1": f1,
        "em": em,
    }
