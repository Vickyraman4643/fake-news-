
##  Fake News Detection using Machine Learning

This project focuses on detecting **fake news articles** using machine learning models trained on labeled datasets containing real and fake news. It employs natural language processing (NLP) and scikit-learn pipelines to classify news as either real or fake.

###  Features

- Reads and processes datasets `Fake.csv` and `True.csv`.
- Text preprocessing (removal of punctuation, stopwords, and special characters).
- Data visualization using `matplotlib` and `seaborn`.
- Splits data into training and testing sets.
- Trains machine learning models for classification.
- Evaluates model performance using metrics such as accuracy and classification report.

### 🛠 Tech Stack

- Python 🐍
- pandas
- scikit-learn
- seaborn & matplotlib
- joblib

###  Dataset

Make sure you have the following files in your working directory:

- `Fake.csv`: Dataset containing fake news.
- `True.csv`: Dataset containing true news.

These datasets should contain at least a `text` and `label` column for model training.

###  How to Run

1. Clone the repository or download the notebook.
2. Install required packages:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn joblib
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook fakenews.ipynb
   ```

###  Output

- Accuracy score of the trained model.
- Confusion matrix and classification report.
- Visualizations for data insights and performance.

###  Note

- Make sure the datasets are properly formatted and present in the correct directory.
- The notebook does not currently contain markdown cells. Consider adding explanations and section headers for better readability.
