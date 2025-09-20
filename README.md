# Financial Returns Distribution Analyzer

## 📌 Project Overview
This project is a **statistical tool for financial data analysis**.  
It fits **multiple probability distributions** (Normal, Student’s t, and Skewed) to real asset returns  
using **Maximum Likelihood Estimation (MLE)**, performs **goodness-of-fit tests**,  
and computes important **distribution moments** such as mean, variance, skewness, and kurtosis.  

The analyzer allows comparison of **distribution parameters across different assets**,  
helping in better risk management and decision-making.

---

## 🚀 Features
- Fit asset returns with **Normal, Student’s t, and Skewed distributions**  
- Use **Maximum Likelihood Estimation (MLE)** for parameter estimation  
- Perform **goodness-of-fit tests** (e.g., Kolmogorov-Smirnov, Anderson-Darling)  
- Compute **statistical moments**: mean, variance, skewness, kurtosis  
- Compare fitted distribution parameters across multiple assets  

---

## ⚙️ Technologies Used
- **Python**  
- **NumPy, Pandas, SciPy** (statistics & computation)  
- **Matplotlib, Seaborn** (data visualization)  
- **yFinance** (fetching financial data)  

---

## 📊 Outputs
- Estimated parameters for Normal, Student’s t, and Skewed distributions  
- Goodness-of-fit test results for each distribution  
- Visualization of fitted probability density functions (PDFs) over real data  
- Comparative analysis of distribution parameters across different assets  

---

## 📥 Installation
```bash
git clone https://github.com/your-username/financial-returns-analyzer.git
cd financial-returns-analyzer
pip install -r requirements.txt
