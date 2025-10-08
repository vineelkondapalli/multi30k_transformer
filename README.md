# multi30k_transformer
# End-to-End Transformer for Neural Machine Translation

## Project Summary

This repository contains a complete, from-scratch implementation of the Transformer model from the "Attention Is All You Need" paper, built using PyTorch. The model is trained on the Multi30k dataset to perform German-to-English machine translation, serving as a deep dive into the architecture that powers modern NLP.

The project began with the construction of a modern data pipeline using the **Hugging Face `datasets`** library for robust data handling and a pre-trained **sub-word tokenizer** from the `transformers` library (`Helsinki-NLP/opus-mt-de-en`). This approach solved the out-of-vocabulary problem inherent in word-level tokenization.

The core of the project was building the **Encoder-Decoder** architecture piece by piece. This included a detailed implementation of the key components:
* The **Multi-Head Self-Attention** mechanism.
* Learnable **Positional Encodings** to give the model a sense of sequence order.
* **Residual Connections** and **Layer Normalization** to enable stable training of a deep network.

To achieve high performance, the training process was enhanced with advanced techniques, including **Label Smoothing** to prevent the model from becoming overconfident, **Weight Decay** for regularization, and a **Cosine Annealing** learning rate scheduler.

Finally, for inference, the standard greedy search was replaced by a custom **Beam Search** decoding algorithm to produce more fluent and accurate translations. The project also explored low-level performance optimization by engineering a custom, fused GPU kernel for the feedforward mechanism using **OpenAI Triton**.

## Technologies Used

* PyTorch
* Hugging Face `datasets` & `transformers`
* spaCy
* OpenAI Triton
* Google Colab & A100 GPU

## Usage

1.  Clone the repository:
    ```bash
    git clone [your-repo-link]
    ```
2.  Install the required dependencies. It's recommended to do this in a virtual environment.
    ```bash
    pip install torch torchvision torchdata transformers datasets spacy triton
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    ```
3.  Run the main Jupyter Notebook or Python script to build the data pipeline, train the model, and perform translation inference.
