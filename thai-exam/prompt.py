prompt = """\
Answer the following question by selecting only from the given choices.
Do not provide any additional explanations or commentary.

---

Step:
- Carefully read the question and all the provided choices.
- Identify the choice that best answers the question.
- Respond with: 'The correct answer is (number)'
- Replace 'number' with the corresponding number of the correct choice

---

Example:
Question: What is the capital of France?
Choices: (1) Berlin (2) London (3) Paris (4) Rome
Answer: The correct answer is (3)

---

Question: {}
Choices: {}
Answer:
"""