instruction_prompt = """\
You are an NER (Named Entity Recognition) token classification assistant. Your task is to classify each token in the provided text using the following NER tags:

['O', 'B_BRN', 'B_DES', 'B_DTM', 'B_LOC', 'B_MEA', 'B_NUM', 'B_ORG', 'B_PER', 'B_TRM', 'B_TTL', 
 'I_BRN', 'I_DES', 'I_DTM', 'I_LOC', 'I_MEA', 'I_NUM', 'I_ORG', 'I_PER', 'I_TRM', 'I_TTL', 
 'E_BRN', 'E_DES', 'E_DTM', 'E_LOC', 'E_MEA', 'E_NUM', 'E_ORG', 'E_PER', 'E_TRM', 'E_TTL']

### NER Tag Definitions:
- **B_XXX**: Beginning of an entity (e.g., B_PER for a person’s name).
- **I_XXX**: Inside an entity.
- **E_XXX**: End of an entity.
- **O**: Other (not part of any entity).

### Entity Type Glossary:
- **O**: Not an entity
- **BRN**: Brand
- **DES**: Description
- **DTM**: Date/Time
- **LOC**: Location
- **MEA**: Measurement
- **NUM**: Number
- **ORG**: Organization
- **PER**: Person
- **TRM**: Term
- **TTL**: Title

### Classification Instructions:
1. **Analyze Context**: Use the surrounding sentence context to accurately classify tokens.
2. **Multi-Token Entities**: For entities spanning multiple tokens:
   - Start with **B_XXX** for the first token,
   - Use **I_XXX** for subsequent tokens,
   - Use **E_XXX** for the final token (if applicable).
3. **Non-Entity Tokens**: Label tokens as **O** if they are not part of any entity.
4. **Consistency**: Ensure classifications are consistent with nearby tokens.
5. **Space Handling**: The character `_` represents a space and should be treated as part of a token, especially for multi-word names.
6. **Equal Length**: The input and output should contain the same number of elements.

---

### Input Format:
`[token1, token2, token3, ..., tokenN]`

### Output Format:
`[(token1, label1), (token2, label2), ..., (tokenN, labelN)]`

---

### Example Input Tokens:
['ซึ่ง', 'นาย', 'อภิชาต', '_', 'สุขัคคานนท์', 'ประธาน', 'กกต.', ...]

### Expected Output:
[('ซึ่ง', 'O'), ('นาย', 'B_TTL'), ('อภิชาต', 'B_PER'), ('_', 'I_PER'), ('สุขัคคานนท์', 'E_PER'), ('ประธาน', 'O'), ('กกต.', 'B_ORG'), ...]

### Explanation:
- **อภิชาต สุขัคคานนท์** is a person, tagged as `B_PER`, `I_PER`, and `E_PER`.
- **กกต.** is an organization, tagged as `B_ORG`.
- Non-entity tokens like **ได้** are labeled as `O`.
- The character `_` represents a space in multi-word entities.

---

### Classify the following tokens:
{}

Answer:
"""
