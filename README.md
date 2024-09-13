
# Model Evaluation

## lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness)

lm-eval-harness เป็นเครื่องมือที่ใช้ในการทดสอบประสิทธิภาพของโมเดลภาษา (Language Models หรือ LM) ในงานต่าง ๆ เช่น การตอบคำถาม, การทำนายคำต่อไป, หรือการแปลภาษา โดยเครื่องมือนี้จะช่วยวัดความสามารถของโมเดลในการทำงานตาม task ที่กำหนด และให้ประเมินผลลัพธ์ในรูปแบบของค่า metrices ต่าง ๆ เช่น accuracy หรือ similarity

lm-eval-harness ถูกออกแบบให้ใช้งานง่าย และรองรับการตั้งค่า เช่น ขนาดของ batch, การเลือกว่าจะใช้ CPU หรือ GPU ในการประมวลผล, และการใช้ระบบ cache เพื่อเร่งความเร็วในการ test

โดยสรุป lm-eval-harness เป็นเครื่องมือที่มีประโยชน์สำหรับการประเมินโมเดลภาษาในหลาย ๆ งาน ช่วยให้ Developer เข้าใจถึงประสิทธิภาพของโมเดลในสถานะการณ์สภาพแวดล้อมต่าง ๆ ได้มากขึ้น

### Installation
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
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

## Run Evaluation Scripts

### Custom Pipeline

This section provides scripts for running evaluations on different datasets using pre-trained models. Make sure to follow the instructions for each task to ensure proper setup and execution.

#### 1. ThaiSUM

ThaiSUM is a dataset used for text summarization tasks in the Thai language.

**Instructions:**

- Ensure you have Python and the required libraries installed. If not, install them using:
  ```bash
  pip install transformers datasets
  ```
- Run the evaluation script for the ThaiSUM dataset using the following command:
  ```bash
  python scripts/thaisum/thaisum.py --pretrained "preechanon/mt5-base-thaisum-text-summarization" --dataset "nakhun/thaisum" --split "test"
  ```
- This command will use the pre-trained model `mt5-base-thaisum-text-summarization` and evaluate it on the test split of the `nakhun/thaisum` dataset.

#### 2. LST20

LST20 is a large-scale dataset for Thai NER (Named Entity Recognition) tasks.

**Instructions:**

- Download the LST20 dataset from [AIForThai](https://aiforthai.in.th/corpus.php). You'll need to log in and navigate to the "Corpus" section to find the LST20 Corpus.
- Extract the downloaded `AIFORTHAI-LST20Corpus.tar.gz` file to your desired directory.
- Ensure you have CUDA installed if you plan to run the model on a GPU.
- Run the evaluation script with the following command:
  ```bash
  python scripts/lst20/lst20.py --cuda --pretrained pythainlp/thainer-corpus-v2-base-model --lst20_path AIFORTHAI-LST20Corpus/LST20_Corpus --split test
  ```
- This command will evaluate the `thainer-corpus-v2-base-model` on the test split of the LST20 dataset.

#### 3. Wisesight Sentiment

Wisesight Sentiment is a dataset for sentiment analysis in the Thai language.

**Instructions:**

- Install the `lm-evaluation-harness` package if you haven't already:
  ```bash
  pip install lm-evaluation-harness
  ```
- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks scripts/wisesight_sentiment --device cuda:0
  ```
- Replace `distilgpt2` with your preferred pre-trained model if needed.

### Custom YAML Files with `lm-evaluation-harness`

This section provides scripts for evaluating models on translated datasets using custom YAML files with `lm-evaluation-harness`.

#### 1. Hellaswag_TH

Hellaswag_TH is a translated version of the Hellaswag dataset into Thai.

**Instructions:**

- The dataset is available at [`Patt/HellaSwag_thai`](https://huggingface.co/datasets/Patt/HellaSwag_thai).
- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks scripts/hellaswag --device cuda:0
  ```

#### 2. RTE_TH

RTE_TH is a translated version of the RTE (Recognizing Textual Entailment) dataset into Thai.

**Instructions:**

- The dataset is available at [`Patt/RTE_TH_drop`](https://huggingface.co/datasets/Patt/RTE_TH_drop).
- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks scripts/rte --device cuda:0
  ```

#### 3. Record_TH

Record_TH is a translated version of the ReCoRD (Reading Comprehension with Commonsense Reasoning Dataset) into Thai.

**Instructions:**

- The dataset is available at [`Patt/ReCoRD_TH_drop`](https://huggingface.co/datasets/Patt/ReCoRD_TH_drop).
- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks scripts/record --device cuda:0
  ```

#### 4. MultiRC_TH

MultiRC_TH is a translated version of the MultiRC (Multi-Sentence Reading Comprehension) dataset into Thai.

**Instructions:**

- The dataset is available at [`Patt/MultiRC_TH_drop`](https://huggingface.co/datasets/Patt/MultiRC_TH_drop).
- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks scripts/multirc --device cuda:0
  ```

#### 5. XQuad_TH

XQuad_TH is a translated subset of the XQuad dataset into Thai, focusing on question answering.

**Instructions:**

- The subset is available from the `google/xquad` dataset on Hugging Face, specifically `xquad.th`.
- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks scripts/xquad_th --device cuda:0
  ```

### Natively Supported by `lm-evaluation-harness`

These tasks are natively supported by the `lm-evaluation-harness` tool and do not require custom scripts.

#### 1. XCopa (TH)

XCopa is a cross-linguistic version of the Choice of Plausible Alternatives (COPA) dataset.

**Instructions:**

- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks xcopa_th --device cuda:0
  ```

#### 2. XNLI (TH)

XNLI is a cross-linguistic version of the NLI (Natural Language Inference) dataset.

**Instructions:**

- Run the evaluation using the following command:
  ```bash
  lm_eval --model hf --model_args pretrained=distilgpt2 --tasks xnli_th --device cuda:0
  ```

### Thai Exam
- We forked from `OpenThaiGPT/openthaigpt_eval` so that will further customize to add prompts
- read this:
