# QLORA
# 🚀 Fine-Tuning Llama 3.2 1B for Price Prediction using QLoRA

This repository contains the complete script and methodology for fine-tuning the `meta-llama/Llama-3.2-1B` model on a custom price prediction task. The primary goal is to train the model to accurately determine the price of a product given its descriptive text.

The project leverages cutting-edge techniques like **QLoRA** for highly efficient training on consumer-grade hardware (2x T4 GPUs on Kaggle) and integrates with **Weights & Biases** for robust experiment tracking.

## ✨ Key Features

-   **🧠 Model:** Utilizes the powerful and compact `meta-llama/Llama-3.2-1B` model as the base.
-   **⚡ Efficient Fine-Tuning:** Employs **QLoRA (Quantized Low-Rank Adaptation)** to drastically reduce memory footprint, enabling fine-tuning on limited VRAM.
-   **💾 4-bit Quantization:** Uses `bitsandbytes` for 4-bit NormalFloat (NF4) quantization, making the model lighter without significant performance degradation.
-   **🎯 Task-Specific Training:** Uses `DataCollatorForCompletionOnlyLM` to focus training solely on the price output (e.g., `Price is $...`), leading to faster and more effective learning.
-   **📈 Experiment Tracking:** Integrated with **Weights & Biases (`wandb`)** to log metrics, hyperparameters, and training progress in real-time.
-   **☁️ Hugging Face Hub Integration:** Automatically saves and pushes the fine-tuned LoRA adapters to the Hugging Face Hub.

## 🛠️ Tech Stack & Libraries

-   **Core Framework:** [PyTorch](https://pytorch.org/)
-   **Model & Tokenizer:** [Hugging Face Transformers](https://huggingface.co/docs/transformers) `v4.43.3`
-   **PEFT (Parameter-Efficient Fine-Tuning):** [Hugging Face PEFT](https://huggingface.co/docs/peft) `v0.11.1` for LoRA implementation.
-   **Training Orchestration:** [Hugging Face TRL](https://huggingface.co/docs/trl) `v0.9.6` (SFTTrainer).
-   **Quantization:** [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) `v0.43.1`.
-   **Distributed Training:** [Hugging Face Accelerate](https://huggingface.co/docs/accelerate) `v0.33.0`.
-   **Experiment Tracking:** [Weights & Biases](https://wandb.ai/) `v0.16.6`.

## 📊 Dataset

The model is fine-tuned on the `ed-donner/pricer-data` dataset from the Hugging Face Hub.

-   The dataset consists of pairs of product descriptions and their corresponding prices.
-   The training script formats the data into a prompt-completion structure, where the model learns to generate the price based on the text. The `DataCollatorForCompletionOnlyLM` is configured with the response template `Price is $` to ensure the model only learns to predict the value that follows.

## 🧠 Fine-Tuning with QLoRA

**QLoRA** is a highly efficient fine-tuning technique that combines:
1.  **4-bit Quantization:** The pre-trained model's weights are loaded and stored in 4-bit precision, which dramatically reduces the memory required.
2.  **Low-Rank Adapters (LoRA):** Small, trainable matrices (adapters) are injected into the model's architecture (specifically the `q_proj`, `k_proj`, `v_proj`, and `o_proj` layers in this project).
3.  **Training:** During fine-tuning, the original model weights remain frozen, and only the lightweight LoRA adapter weights are updated.

This approach allows for fine-tuning large models on a fraction of the hardware typically required, democratizing access to state-of-the-art LLMs.

## ⚙️ Hyperparameter Configuration

The script is configured with the following hyperparameters, which can be easily modified.

### LoRA Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `LORA_A` (Rank) | `256` | The rank of the LoRA matrices. Higher rank means more trainable parameters. |
| `LORA_ALPHA` | `512` | The scaling factor for the LoRA weights. |
| `LORA_DROPOUT` | `0.1` | Dropout probability for the LoRA layers to prevent overfitting. |
| `TARGET_MODULES`| `q_proj`, `v_proj`, `k_proj`, `o_proj` | The model layers to which LoRA adapters are applied. |

### Training Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `EPOCHS` | `3` | Number of complete passes through the training data. |
| `BATCH_SIZE` | `32` | Number of training examples per device per batch. |
| `LEARNING_RATE` | `1e-4` | The initial learning rate for the AdamW optimizer. |
| `OPTIMIZER` | `paged_adamw_32bit`| Memory-efficient optimizer suitable for quantized models. |
| `LR_SCHEDULER_TYPE` | `cosine` | Learning rate schedule to adjust LR during training. |
| `WARMUP_RATIO` | `0.03` | Percentage of training steps used for a linear warmup. |
| `MAX_SEQUENCE_LENGTH` | `182` | Maximum token length for model inputs. |

## 🚀 How to Run

1.  **Clone the Repository:**

2.  **Install Dependencies:**
    The script includes the necessary `pip install` commands at the top. Ensure you are in a compatible environment (e.g., Kaggle Notebook, Google Colab with GPU).

3.  **Set Up Secrets 🔑:**
    Before running, you need to provide your access tokens. Open the script and replace the placeholder values:
    -   `HF_TOKEN`: Your Hugging Face token with `write` permissions to push the model to the Hub.
    -   `WANDBI_TOKEN`: Your Weights & Biases API key for experiment tracking.

    **Note:** For public repositories, it is highly recommended to use environment variables or a secrets management tool (like Kaggle Secrets) instead of hardcoding tokens in the script.

4.  **Execute the Script:**
    Run the Python script in your environment. The `main()` function will handle:
    -   Logging into Hugging Face and `wandb`.
    -   Loading the dataset and tokenizer.
    -   Configuring the model with 4-bit quantization.
    -   Setting up the `SFTTrainer` with LoRA and training arguments.
    -   Starting the fine-tuning process.

5.  **Monitor & Access Results:**
    -   Follow the training progress in real-time through the link provided in your console for your **Weights & Biases dashboard**.
    -   Once training is complete, the fine-tuned LoRA adapters will be available in your private repository on the **Hugging Face Hub**.
