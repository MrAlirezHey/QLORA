# -*- coding: utf-8 -*-


!pip install -U transformers==4.43.3 \
               accelerate==0.33.0 \
               bitsandbytes==0.43.1 \
               peft==0.11.1 \
               trl==0.9.6 \
               wandb==0.16.6 \
               triton==2.1.0

!pip uninstall -y tensorflow

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import os
from datetime import datetime
from accelerate import notebook_launcher
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
from kaggle_secrets import UserSecretsClient
from accelerate import notebook_launcher
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def main():
    import os
    import re
    import math
    from tqdm import tqdm
    from google.colab import userdata
    from huggingface_hub import login
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig,TrainingArguments
    from datasets import load_dataset, Dataset, DatasetDict
    import wandb
    from peft import LoraConfig
    from trl import SFTTrainer
    from datetime import datetime
    import matplotlib.pyplot as plt
    import torch.nn as nn
    from trl import DataCollatorForCompletionOnlyLM
    torch.set_num_threads(1)
    BASE_MODEL='meta-llama/Llama-3.2-1B'
    PROJECT_NAME='pricer'
    HF_USER='Alireza0017'
    DATASET_NAME='ed-donner/pricer-data'
    RUN_NAME=f'{datetime.now():%Y-%m-%d_%H.%M.%s}'
    PROJECT_RUN_NAME=f'{PROJECT_NAME}-{RUN_NAME}'
    HUB_MODEL_NAME=f'{HF_USER}/{PROJECT_RUN_NAME}'
    MAX_SEQUENCE_LENGTH=182
    #######HyperParameters LORA########
    LORA_A=256
    LORA_ALPHA=512
    TARGET_MODULES=['q_proj','v_proj','k_proj','o_proj']
    LORA_DROPOUT=.1
    QUANT_4BIT=True
    ######HyperParameters Training######
    EPOCHS=3
    BATCH_SIZE=32
    GRADIENT_ACCUMULATION_STEPS=1
    LEARNING_RATE=1e-4
    LR_SCHEDULER_TYPE='cosine'
    WARMUP_RATIO=.03
    OPTIMIZER='paged_adamw_32bit'
    STEPS=30
    SAVE_STEPS=2000
    LOG_TO_WANDB=True

    ###########HF LOGIN############
    HF_TOKEN='**************************'
    login(HF_TOKEN)
    ##########WANDB LOGIN###########
    WANDBI_TOKEN='***********************'
    wandb.login(key=WANDBI_TOKEN)
    ##########DATASET###############
    dataset=load_dataset(DATASET_NAME)
    train=dataset['train']
    test=dataset['test']
    train=train.select(range(20000))
    #########WANDBI INIT###############
    if LOG_TO_WANDB:
        wandb.init(project=PROJECT_NAME,name=RUN_NAME)
    if QUANT_4BIT:
        quant_config= BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
        )
    else:
        quant_config=BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
    device='cuda'
    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL , trust_remote_code=True)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    base_model=AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                             quantization_config=quant_config,device_map="auto"
      )
    base_model.generation_config.pad_token_id=tokenizer.pad_token_id
    #base_model=nn.DataParallel(base_model)
    #base_model=base_model.to(device)
    respose_template="Price is $"
    collator=DataCollatorForCompletionOnlyLM(respose_template,tokenizer=tokenizer)
    ######TRAIN######
    lora_prameters=LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_A,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=TARGET_MODULES
    )
    train_parameters=TrainingArguments(   output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    eval_strategy="steps",
    eval_steps=1000,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else None,
    run_name=RUN_NAME,
    dataloader_num_workers=0,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True)
    fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train,
    eval_dataset=test,
    peft_config=lora_prameters,
    args=train_parameters,
    data_collator=collator,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    dataset_text_field="text",
      )
    fine_tuning.train()
#notebook_launcher(main,num_processes=2)
main()