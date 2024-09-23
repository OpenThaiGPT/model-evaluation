template = """\
Premise: {}
Hypothesis: {}
Is a given Hypothesis entailmented from premise. Answer only True or False?
Answer:
"""


def doc_to_text(doc):
    return template.format(doc["premise"], doc["hypothesis"])


def doc_to_target(doc):
    return doc["label"]


def doc_to_choice(doc):
    return ["True", "False"]
