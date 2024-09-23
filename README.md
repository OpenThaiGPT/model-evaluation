
# Model Evaluation

# Run Evaluation Scripts


This section provides scripts for running evaluations on different datasets using pre-trained models. Make sure to follow the instructions for each task to ensure proper setup and execution.

There is a reuqiredments to install:

```Python
git clone https://github.com/OpenThaiGPT/model-evaluation.git
pip install -e .
```

## 1. ThaiSUM

- ThaiSUM is a dataset used for text summarization tasks in the Thai language.
- This will evaluated using ROUGE, BLEU score and used `attacut` to tokenize
- When running evaluation successfully. `rouge_score.json` and `bleu_score.json` will be generated in directory.
- Run using this code:
```bash
script/thaisum/thaisum.py --model <model> --thaisum_path <path> --split <split> --limit <limit>
```
- `model`: supported models can be seen at `src/model_evaluation/models`
- `thaisum_path`: a path of dataset (HF format)
- `split`: a subset of datasets
- `limit`: to sample element in a given dataset

**You can call API by the following steps**
- Go to directory: `cd script/thaisum`
- Edit `__name__ == "__main__"` section
- Some example code:
```Python
from model_evaluation.models.openthaigpt_hf_7b_2023 import OpenThaiGPTHF7B20234

model = OpenThaiGPTHF7B2023(model_name="openthaigpt/openthaigpt-1.0.0-7b-chat")
thaisum(model, thaisum_path="nakhun/thaisum", split="test", limit=20)
```

## 2. LST20

- LST20 is a large-scale dataset for Thai NER (Named Entity Recognition) tasks.
- **Download LST20** from [AIForThai](https://aiforthai.in.th/corpus.php).
  - Extract the downloaded `AIFORTHAI-LST20Corpus.tar.gz` file to your desired directory.
- Run using this code:
```bash
python scripts/lst20/lst20.py --model <model> --lst20_path "path/to/AIFORTHAI-LST20Corpus/LST20_Corpus" --split <split> --limit <limit>
```
- `model`: supported models can be seen at `src/model_evaluation/models`
- `lst20_path`: a path of dataset (HF format) loaded from AiForThai
- `split`: a subset of datasets
- `limit`: to sample element in a given dataset
  
## lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness)

lm-eval-harness เป็นเครื่องมือที่ใช้ในการทดสอบประสิทธิภาพของโมเดลภาษา (Language Models หรือ LM) ในงานต่าง ๆ เช่น การตอบคำถาม, การทำนายคำต่อไป, หรือการแปลภาษา โดยเครื่องมือนี้จะช่วยวัดความสามารถของโมเดลในการทำงานตาม task ที่กำหนด และให้ประเมินผลลัพธ์ในรูปแบบของค่า metrices ต่าง ๆ เช่น accuracy หรือ similarity

lm-eval-harness ถูกออกแบบให้ใช้งานง่าย และรองรับการตั้งค่า เช่น ขนาดของ batch, การเลือกว่าจะใช้ CPU หรือ GPU ในการประมวลผล, และการใช้ระบบ cache เพื่อเร่งความเร็วในการ test

โดยสรุป lm-eval-harness เป็นเครื่องมือที่มีประโยชน์สำหรับการประเมินโมเดลภาษาในหลาย ๆ งาน ช่วยให้ Developer เข้าใจถึงประสิทธิภาพของโมเดลในสถานะการณ์สภาพแวดล้อมต่าง ๆ ได้มากขึ้น

### Installation

To use GPT, requiring CUDA-supported torch download at https://pytorch.org/get-started/locally

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
pip install -e lm-evaluation-harness
```

### Usage

| **Argument**             | **Short Option** | **Type** | **Default Value**    | **Description**                                                                                                                                                                 |
| ------------------------ | ---------------- | -------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model`                | `-m`             | `str`    | `"hf"`               | ชื่อโมเดล เช่น `"hf"`                                                                                                                                                           |
| `--tasks`                | `-t`             | `str`    | `None`               | list ของ task ที่ต้องการ evaluation format string เช่น `task1,task2` ใช้คำสั่ง `lm-eval --tasks {list_groups,list_subtasks,list_tags,list}` เพื่อแสดง task ทั้งหมดใน default |
| `--model_args`           | `-a`             | `str`    | `""`                 | Parameter ของโมเดลใน format string เช่น `pretrained=EleutherAI/pythia-160m,dtype=float32`                                                                                  |
| `--num_fewshot`          | `-f`             | `int`    | `None`               | จำนวนที่จะทำ few-shot                                                                                                                                               |
| `--batch_size`           | `-b`             | `str`    | `1`                  | ขนาด batch มี option `"auto"`, `"auto:N"` หรือ `N` (ทั้งหมด) ค่า default คือ 1                                                                                                     |
| `--max_batch_size`       | `None`           | `int`    | `None`               | ขนาด max batch size config ร่วมกับ `--batch_size auto`                                                                                                                                   |
| `--device`               | `None`           | `str`    | `None`               | `device` ที่จะใช้ (เช่น `cuda`, `cuda:0`, `cpu`)                                                                                                                                          |
| `--output_path`          | `-o`             | `str`    | `None`               | `path` ไปยังไฟล์ output                                                               |
| `--limit`                | `-L`             | `float`  | `None`               | limit จำนวนตัวอย่างต่อ task ถ้าน้อยกว่า 1 = จำนวนตัวอย่างทั้งหมด                                                                                                               |
| `--use_cache`            | `-c`             | `str`    | `None`               | ใช้ cache (sqlite) deault `"None"` หากไม่ต้องการแคช                                                                                             |
| `--cache_requests`       | `None`           | `str`    | `None`               | cache request มี option ค่า `"true"`, `"refresh"`, `"delete"`                                                                                                        |
| `--check_integrity`      | `None`           | `bool`   | `False`              | ตรวจสอบความถูหต้องของ task ที่เกี่ยวข้องหรือไม่                                                                                                                                   |
| `--write_out`            | `-w`             | `bool`   | `False`              | กำหนด prompt                                                                                                                                                      |
| `--log_samples`          | `-s`             | `bool`   | `False`              | if True จะ save output ของโมเดลและ document สำหรับ evaluation example ใช้ร่วมกับ `--output_path`                                                                                   |
| `--system_instruction`   | `None`           | `str`    | `None`               | กำหนด `system_instruction prompt`                                                                                                                                                          |
| `--apply_chat_template`  | `None`           | `str`    | `False`              | If True ใช้ apply_chat_template จะใช้แม่แบบการแชทเริ่มต้น                                                                                                                     |
| `--fewshot_as_multiturn` | `None`           | `bool`   | `False`              | If True ใช้ fewshot เป็นการสนทนาแบบหลายครั้ง                                                                                                                                          |
| `--show_config`          | `None`           | `bool`   | `False`              | If True แสดงการตั้งค่าทั้งหมดของ task ที่ท้ายการประเมิน                                                                                                                                    |
| `--include_path`         | `None`           | `str`    | `None`               | เพิ่ม path task ภายนอกที่จะรวมเข้ามา                                                                                                                                         |
| `--gen_kwargs`           | `None`           | `str`    | `None`               | สำหรับสร้างโมเดลใน greedy_until tasks เช่น `temperature=0,top_k=0,top_p=0`                                                                                              |
| `--verbosity`            | `-v`             | `str`    | `"INFO"`             | logging สำหรับ debug ค่า `"CRITICAL"`, `"ERROR"`, `"WARNING"`, `"INFO"`, `"DEBUG"`                                                                         |
| `--wandb_args`           | `None`           | `str`    | `""`                 | สำหรับเก็บผลลง `wandb` โดยอ่านจาก `wandb.init` เช่น `project=lm-eval,job_type=eval`                                                                                          |
| `--hf_hub_log_args`      | `None`           | `str`    | `""`                 | สำหรับเก็บผลลง `Hugging Face` เช่น `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`                                            |
| `--predict_only`         | `-x`             | `bool`   | `False`              | ใช้ร่วมกับ `--log_samples` บันทึกเฉพาะผลลัพธ์ของโมเดลและจะไม่ประเมินค่ามาตรวัด                                                                                                             |
| `--seed`                 | `None`           | `str`    | `"0,1234,1234,1234"` | กำหนด seed สำหรับ `random`, `numpy`, `torch`, และการสุ่มตัวอย่าง fewshot ค่าเริ่มต้นคือ `"0,1234,1234,1234"`                  |
| `--trust_remote_code`    | `None`           | `bool`   | `False`              | `trust_remote_code` เป็น `True` เพื่อสร้าง HF Datasets จาก Hub                                                                                                |

## Custom YAML Files with `lm-evaluation-harness`

This section provides scripts for evaluating models on translated datasets using custom YAML files with `lm-evaluation-harness`.

### Copy the config into `lm-evaluation-harness\lm_eval\tasks` then run it directly. Errors may occured when run outside of `lm-evaluation-harness\lm_eval\tasks` 

### 1. Hellaswag_TH

- Hellaswag_TH is a translated version of the Hellaswag dataset into Thai.
- The dataset is available at [`Patt/HellaSwag_thai`](https://huggingface.co/datasets/Patt/HellaSwag_thai).
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks hellaswag_th
```

### 2. RTE_TH

- RTE_TH is a translated version of the RTE (Recognizing Textual Entailment) dataset into Thai.
- The dataset is available at [`Patt/RTE_TH_drop`](https://huggingface.co/datasets/Patt/RTE_TH_drop).
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks rte_th
```

### 3. Record_TH

- Record_TH is a translated version of the ReCoRD (Reading Comprehension with Commonsense Reasoning Dataset) into Thai.
- The dataset is available at [`Patt/ReCoRD_TH_drop`](https://huggingface.co/datasets/Patt/ReCoRD_TH_drop).
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks record_th
```

### 4. MultiRC_TH

- MultiRC_TH is a translated version of the MultiRC (Multi-Sentence Reading Comprehension) dataset into Thai.
- The dataset is available at [`Patt/MultiRC_TH_drop`](https://huggingface.co/datasets/Patt/MultiRC_TH_drop).
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks multirc_th
```

### 5. XQuad_TH

XQuad_TH is a translated subset of the XQuad dataset into Thai, focusing on question answering.
- The subset is available from the `google/xquad` dataset on Hugging Face, specifically `xquad.th`.
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks xquad_th
```

### 6. Wisesight Sentiment

- Wisesight Sentiment is a dataset for sentiment analysis in the Thai language.
- The dataset is available at [`pythainlp/wisesight_sentiment`](https://huggingface.co/datasets/pythainlp/wisesight_sentiment).
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks wisesight_sentiment
```

### Natively Supported by `lm-evaluation-harness`

These tasks are natively supported by the `lm-evaluation-harness` tool and do not require custom scripts.

### 1. XCopa (TH)

- XCopa is a cross-linguistic version of the Choice of Plausible Alternatives (COPA) dataset.
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks xcopa_th
```

### 2. XNLI (TH)

- XNLI is a cross-linguistic version of the NLI (Natural Language Inference) dataset.
- Run the evaluation using the following command:
```bash
lm_eval --model hf --model_args "pretrained=<pretrained>" --tasks xnli_th
```


## Thai Exam
- We forked from `OpenThaiGPT/openthaigpt_eval` so that will further customize to add prompts
- We have added exam data (csv): A-Level, TPAT-1, TGAT from [thai_exam](https://huggingface.co/datasets/scb10x/thai_exam)
- Read this for details [Here](https://github.com/OpenThaiGPT/model-evaluation/tree/main/thai-exam/exams)

## Notes

- Torch with CUDA is required.
  - `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

- Accerlate from HF also required when using `device_map`.
  - `pip install git+https://github.com/huggingface/accelerate`