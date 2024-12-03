# leaderboard|mmlu:machine_learning

- The **MMLU: Machine Learning (ML)** dataset is part of the Massive Multitask Language Understanding (MMLU) benchmark.
- It uses a multiple-choice question format, where each question includes four answer options, and only one is correct.
- **Scoring:** Performance is measured as the percentage of questions answered correctly. A random baseline would achieve ~25%, as each question has four options.

    ```
    Example: Dimensionality Reduction

    Which technique is primarily used to reduce the dimensionality of data while preserving as much variance as possible?

    A) Linear Discriminant Analysis (LDA)
    B) Singular Value Decomposition (SVD)
    C) Principal Component Analysis (PCA)
    D) K-Nearest Neighbors (KNN)

    Answer: C) Principal Component Analysis (PCA)
    ```

## GPT2
- ### Unquantized
    `"model_size": "476.2 MB"`

    Runtime: 6 seconds
    ```
    |               Task                |Version|Metric|Value |   |Stderr|
    |-----------------------------------|------:|------|-----:|---|-----:|
    |all                                |       |acc   |0.3304|±  |0.0446|
    |leaderboard:mmlu:machine_learning:0|      0|acc   |0.3304|±  |0.0446|
    ```

## Ministral-8B-Instruct-2410
- ### Unquantized
    `"model_size": "14.94 GB"`

    Runtime: 441 seconds

    ```
    |               Task                |Version|Metric|Value |   |Stderr|
    |-----------------------------------|------:|------|-----:|---|-----:|
    |all                                |       |acc   |0.4643|±  |0.0473|
    |leaderboard:mmlu:machine_learning:0|      0|acc   |0.4643|±  |0.0473|
    ```
- ### 4-bit quantized
    `"model_size": "5.23 GB"`

    Runtime: 57 seconds

    ```
    |               Task                |Version|Metric|Value |   |Stderr|
    |-----------------------------------|------:|------|-----:|---|-----:|
    |all                                |       |acc   |0.4018|±  |0.0465|
    |leaderboard:mmlu:machine_learning:0|      0|acc   |0.4018|±  |0.0465|
    ```

## Mistral-7B-Instruct-v0.3
- ### Unquantized
    `"model_size": "13.5 GB"`

    Runtime: 397 seconds

    ```
    |               Task                |Version|Metric|Value |   |Stderr|
    |-----------------------------------|------:|------|-----:|---|-----:|
    |all                                |       |acc   |0.5446|±  |0.0473|
    |leaderboard:mmlu:machine_learning:0|      0|acc   |0.5446|±  |0.0473|
    ```
- ### 4-bit quantized
    ` "model_size": "3.75 GB"`

    Runtime: 58.3 seconds

    ```
    |               Task                |Version|Metric|Value |   |Stderr|
    |-----------------------------------|------:|------|-----:|---|-----:|
    |all                                |       |acc   |0.4732|±  |0.0474|
    |leaderboard:mmlu:machine_learning:0|      0|acc   |0.4732|±  |0.0474|
    ```

## Mistral-7B-Instruct-v0.2
- ### Unquantized
    `"model_size": "13.49 GB"`

    Runtime: 407 seconds

        ```
        |               Task                |Version|Metric|Value |   |Stderr|
        |-----------------------------------|------:|------|-----:|---|-----:|
        |all                                |       |acc   |0.5089|±  |0.0475|
        |leaderboard:mmlu:machine_learning:0|      0|acc   |0.5089|±  |0.0475|
        ```

- ### 4-bit quantized
    `"model_size": "3.74 GB"`
    Runtime: 54 seconds

        ```
        |               Task                |Version|Metric|Value |   |Stderr|
        |-----------------------------------|------:|------|-----:|---|-----:|
        |all                                |       |acc   |0.4821|±  |0.0474|
        |leaderboard:mmlu:machine_learning:0|      0|acc   |0.4821|±  |0.0474|
        ```

## Mistral-7B-v0.3
- ### Unquantized
    `"model_size": "13.5 GB"`
    Runtime: 396 seconds

        ```
        |               Task                |Version|Metric|Value |   |Stderr|
        |-----------------------------------|------:|------|-----:|---|-----:|
        |all                                |       |acc   |0.4643|±  |0.0473|
        |leaderboard:mmlu:machine_learning:0|      0|acc   |0.4643|±  |0.0473|
        ```

- ### Finetune: `cognitivecomputations/dolphin-2.9.3-mistral-7B-32k`
    Runtime: 398 seconds

        ```
        |               Task                |Version|Metric|Value |   |Stderr|
        |-----------------------------------|------:|------|-----:|---|-----:|
        |all                                |       |acc   |0.3661|±  |0.0457|
        |leaderboard:mmlu:machine_learning:0|      0|acc   |0.3661|±  |0.0457|
        ```