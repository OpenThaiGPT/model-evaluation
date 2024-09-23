import re
import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace("[ส่วนหัว]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx_a_th = doc["ctx_a_th"] if doc["ctx_a_th"] is not None else ""
        ctx_b_th = doc["ctx_b_th"] if doc["ctx_b_th"] is not None else ""
        ctx = ctx_a_th + " " + ctx_b_th
        doc["endings_th"] = doc["endings_th"].split(",")

        out_doc = {
            "query": preprocess(doc["activity_label_th"] + ": " + ctx),
            "choices": [preprocess(ending)[:1024] for ending in doc["endings_th"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
