import datasets

def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        return {
            "query": "What is the sentiment of this sentence: "
            + doc["texts"]
            + "\nAnswer:",
            "choices": ["Positive", "Neutral", "Negative"],
            "gold": doc["category"],
        }

    return dataset.map(_process_doc)