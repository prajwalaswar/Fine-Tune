# 📌 Medical QA Dataset

## 📖 Overview
This dataset contains a collection of medical-related questions and their corresponding answers. The dataset is designed to assist in training AI models for medical question-answering tasks. It covers a wide range of topics, including general health, diseases, treatments, medications, and medical procedures.

## 📂 Dataset Format
The dataset is stored in a `.jsonl` (JSON Lines) format, where each entry consists of a **prompt** (question) and a **completion** (answer).

### 🔹 Example Data Entry:
```json
{
    "prompt": "Question 1: What are the common symptoms of diabetes?",
    "completion": "Answer 1: Common symptoms of diabetes include frequent urination, excessive thirst, unexplained weight loss, fatigue, and blurred vision."
}
```

## 🎯 Usage
This dataset can be used for:
- Fine-tuning language models for medical Q&A tasks.
- Building AI-powered chatbots for healthcare assistance.
- Research on medical question-answering systems.

## 🚀 How to Use in Google Colab
1. **Upload the dataset** to Google Colab.
2. **Load the dataset** using the following Python code:
    ```python
    import json

    dataset_path = "/content/medical_dataset.jsonl"
    dataset = []
    with open(dataset_path, "r") as file:
        for line in file:
            dataset.append(json.loads(line))
    ```
3. **Access the data**:
    ```python
    for entry in dataset[:5]:
        print("Q:", entry["prompt"])
        print("A:", entry["completion"])
        print()
    ```

## 🛠 Requirements
- Python 3.x
- Google Colab or Jupyter Notebook
- `json` module for data handling

## 💡 Contribution
Feel free to contribute by adding more medical questions, improving answers, or enhancing the dataset for better AI performance.

## 📜 License
This dataset is intended for educational and research purposes only. Always consult medical professionals for health-related concerns.

---
**🌟 Happy Learning! 🚀**

