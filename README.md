# Titanic Survival Prediction using Decision Trees 🚢🌳

This project applies a **Decision Tree classifier** to the Titanic dataset for predicting passenger survival.  
It includes data preprocessing, evaluation metrics, cross-validation, and visualization using Graphviz.  

---

## ✨ My Learning Journey
I started learning some **Machine Learning concepts** through SoloLearn, and this project is one of my learnings on **Decision Trees**.  

The inspiration and base idea came from SoloLearn. When I began working on this, I was still exploring the concepts that would be applied here — like basic **pandas** for data handling and **Graphviz** for tree visualization (I’ll deep dive into them later, since I believe in *learning by doing* 😅).  

For this project, I used the [SoloLearn Titanic dataset](https://sololearn.com/uploads/files/titanic.csv), since it doesn’t contain any missing values.  

I also learned some **basic evaluation techniques** like accuracy, precision, and recall, and applied them here.  

In the future, I plan to add **pre-pruning techniques** to handle overfitting, because Decision Trees are usually prone to it.  

---

## 📊 Dataset
- Source: [SoloLearn Titanic dataset](https://sololearn.com/uploads/files/titanic.csv)  
- Features used: `Pclass`, `Sex`, `Age`, `Siblings/Spouses`, `Parents/Children`, `Fare`  
- Target: `Survived` (0 = Not Survived, 1 = Survived)  

---

## 🛠 Installation

Clone the repository:

git clone https://github.com/Girish-Kumar-Sahu/titanic-decision-tree.git
cd titanic-decision-tree
Install dependencies:
pip install -r requirements.txt

---
▶️ How to Run
Run the script:


python titanic_decision_tree.py
This will:

Train Decision Tree models with both gini and entropy criteria

Perform 5-fold cross-validation

Print accuracy, precision, and recall

Export a decision tree visualization as tree.png

📷 Example Output
makefile
Copy code
Decision Tree - gini
accuracy: 0.78
precision: 0.72
recall: 0.68

Decision Tree - entropy
accuracy: 0.79
precision: 0.73
recall: 0.69

---

🔮 Future Improvements
Implement pre-pruning (max depth, min samples split)

Add post-pruning methods

Try other ML models (Random Forest, Logistic Regression)

Convert into a Jupyter Notebook with visualizations

🙏 Acknowledgements
SoloLearn for dataset and initial guidance.

scikit-learn & Graphviz documentation for implementation details.