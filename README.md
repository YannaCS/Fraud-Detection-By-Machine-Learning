# Fraud Detection By Machine Learning

A comprehensive machine learning project for detecting fraudulent transactions using various ML algorithms and techniques.

## 📋 Overview

This repository contains a complete fraud detection pipeline implemented in Python, featuring exploratory data analysis, feature engineering, and multiple modeling approaches including automated imbalance handling techniques.


## 📁 Project Structure

The project consists of four main Jupyter notebooks that should be explored in the following order:

1. **EDA.ipynb** - Exploratory Data Analysis
2. **feature_engineering.ipynb** - Feature Engineering and Preprocessing
3. **modeling.ipynb** - Model Development and Evaluation
4. **modeling_autoImbalanceHandle.ipynb** - Automated Imbalance Handling Techniques

## 📊 Notebooks Overview

### 1. EDA (Exploratory Data Analysis)
- Initial data exploration and visualization
- Statistical analysis of fraud vs. non-fraud transactions
- Distribution analysis of features
- Identification of patterns and anomalies
- Missing value analysis

### 2. Feature Engineering
- Feature creation and transformation
- Handling categorical variables
- Scaling and normalization techniques
- Feature selection methods
- Dimensionality reduction (if applicable)

### 3. Modeling
- Implementation of various ML algorithms:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural Networks
  - Gradient Boost Decision Tree
  - LightGBM
  - Decision Tree
- Model evaluation metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC
  - Confusion Matrix
- Cross-validation strategies
- Auto select the best performaced model based on weighted score

### 4. Modeling with Auto Imbalance Handling
- Automated techniques for handling class imbalance:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Random Under-sampling
  - Ensemble methods
- Comparison of different balancing techniques
- Performance evaluation on imbalanced datasets

## 🔧 Performance Optimization

### Intel Python Acceleration
For faster model training and processing, you can use Intel Python distribution with JIT compilation:

```python
# Enable Intel Python optimizations
import intel_numpy
import intel_scipy
```

This can significantly speed up numerical computations and model training times.


## 📊 Dataset

The project uses a fraud detection dataset containing:
- Transaction details
- Customer information
- Merchant data
- Transaction amounts and timestamps
- Binary labels (fraud/non-fraud)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📚 References

- DOI: [10.1109/MLBDBI51377.2020.00025](https://doi.org/10.1109/MLBDBI51377.2020.00025)
- Related research papers and methodologies used in this project

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- YannaCS 
- [Personal Page](https://yannacs.github.io/)
- [GitHub Profile](https://github.com/YannaCS)

## 🔄 Updates

- **17/09/2020**: Intel Python and JIT compilation support added for performance optimization
- **10/06/2025**: Organize the files and move functions to .py files

## 📞 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: Make sure to follow the suggested notebook order (EDA → Feature Engineering → Modeling → Modeling with Auto Imbalance Handle) for the best understanding of the fraud detection pipeline.

