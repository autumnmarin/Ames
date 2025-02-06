# 🏡 The Real Estate Crystal Ball: Predicting Property Values with Smarter Data  

## 📌 TL;DR – The Non-Technical Summary  
Michael Lewis once said that he always writes to one specific person—someone real, someone he wants to impress, someone who will challenge him if the work doesn’t hold up.  

**RP, this project is for you.**  

If you ask a **real estate professional** what determines a property's value, they won’t just point to square footage or the number of units. They’ll think about **the whole picture**—location, condition, land use, and how all the pieces fit together. But here’s the thing: most models that predict property values don’t work that way.  

Most data science projects on the **Ames housing dataset**—which is real-world assessor data from **Ames, Iowa**—take the "biggest number wins" approach. They only look at which individual features have the highest correlation with sale price, ignoring **how features interact**—which is exactly how a human would think about it.  

### **This project fixes that.**  

By applying **real estate knowledge to feature engineering**, I tested whether a machine learning model could make better predictions by thinking more like an appraiser. The result?  
- ✅ **Traditional models (like linear regression) got a boost** from this approach.  
- ✅ **More advanced models (like Gradient Boosting) barely changed**—they were unimproved.  

### **Key Takeaway:**  
> **When you help a simpler model "see" the relationships between property features, it performs better. But the smartest models already figure that out on their own.**  

While this study focused on residential properties, the approach could extend to commercial real estate.  

And ultimately, the **real strength of this analysis lies in the underlying mathematics—** the ability to structure, transform, and extract meaningful insights from data.

Oh, and one more thing: **this is real-world, ugly data.** It’s full of missing values, inconsistencies, and weird edge cases—just like any dataset you’d find in commercial real estate. Cleaning it up and making it usable was half the battle.  

---

## 🎯 Project Objectives  
1️⃣ **Reframe Feature Engineering with Domain Knowledge**  
   - Instead of focusing only on the highest-correlated features, variables were grouped based on real estate logic.  

2️⃣ **Compare Model Performance on Two Tracks**  
   - **Basic Processed Version:** Standard data preprocessing (handling missing values, standardization, one-hot encoding).  
   - **Highly Feature-Engineered Version:** Grouped and transformed features based on real estate knowledge.  

3️⃣ **Evaluate the Impact of Feature Engineering Across Models**  
   - 🏠 **Linear Regression (Elastic Net)**  
   - 🌲 **Decision Tree**  
   - 🌳 **Random Forest**  
   - ⚡ **Gradient Boosting (LightGBM)**  
   - 🚀 **CatBoost (Best Model: RMSE 19,875)**  

---

## 🛠 Methodology  

### **🔹 Breaking Down the Features by Category**  
With **over 80 columns** in the dataset, simply looking at correlation charts wasn’t enough. Instead, I broke all features into **major categories** and evaluated them **one category at a time** to understand their impact.  

Each feature was grouped into **logical real estate categories**, including:  
- **Interior Features:** Square footage, HVAC, bedroom count
- **Exterior Features:** Exterior material, driveway, wood deck square footage  
- **Basement Features:** Finished vs. unfinished space, bathrooms  
- **Lot & Land Features:** Lot size, frontage, type
- **Garage Features:** Number of cars, attached vs. detached, year built
- **Sales History & Transaction Details:** Sale type, year

By **analyzing features in groups**, I could see where domain knowledge added value. Instead of just keeping high-correlation variables, I was able to **engineer new features that made sense**—such as combining all porch-related variables into a single metric for "outdoor living space."  

### **🔹 Data Preprocessing**  
I created two data processing tracks:  
1️⃣ **Basic Version:** Applied necessary preprocessing (e.g., handling missing values, encoding categorical variables) without feature engineering.  
2️⃣ **Feature-Engineered Version:** Grouped related features, introduced new variables based on domain knowledge, and tested interactions that reflect real-world property valuation.  

🛠 **Dealing with real-world data** was a major part of this process:  
✅ Handling missing values  
✅ Identifying and removing outliers  
✅ Encoding categorical variables efficiently  

---

## 📊 Model Performance Table  

| Model         | RMSE Before | RMSE After  |
|--------------|------------|------------|
| Decision Tree | **37,033** | **36,282** |
| Linear Regression   | **24,471** | **21,975** |
| Gradient Boosting | **21,995** | **20,250** |
| XG Boost | **21,995** | **20,250** |
| CatBoost     | **22,427** | **19,875** |

Gradient Boosting performed the best overall but responded differently to feature engineering compared to linear regression. This raised questions about how tree-based models interpret new features versus linear models.  

---
## 🔎 Findings  

### **1️⃣ Feature Engineering Improved Linear Regression Significantly, but Had Mixed Effects on Tree-Based Models**  
- The feature-engineered dataset **notably improved Linear Regression**, reducing RMSE from **24,471 → 21,975**. This suggests that traditional models benefit greatly from structured transformations.  
- **Decision Trees showed only a minor improvement** (**37,033 → 36,282**), reinforcing their sensitivity to overfitting.  
- **Gradient Boosting, XGBoost, and CatBoost saw moderate improvements**, but their initial RMSEs were already competitive, indicating these models inherently capture key relationships in the data.  
- **CatBoost performed the best overall**, achieving an RMSE of **19,875** after feature engineering.  

### **2️⃣ Feature Engineering Clarified Important Real Estate Trends**  
- Instead of analyzing each feature in isolation, grouping them revealed deeper insights into how home characteristics interact, providing a structured and methodical approach to exploratory data analysis (EDA).
- For example, **combining porch-related variables** into a single outdoor space metric improved interpretability and correlation with home value.  

### **3️⃣ Model Stacking & Future Improvements**  
- **Model stacking** could further enhance accuracy by combining strengths of different models.  
- **Hierarchical models** may better capture complex interactions, such as how **neighborhood influences home values** over time.  
- **Neighborhood significance could be further boosted** by introducing composite features that reflect local price trends and market conditions.  


---

## 🏁 Conclusion  
This project demonstrated that **domain-driven feature engineering** can meaningfully improve traditional models but may have **a limited effect on advanced tree-based models like Gradient Boosting**.  

---

