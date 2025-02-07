![Image](https://github.com/user-attachments/assets/1a2a7b22-77bd-44bf-8567-aa1d5722cd7c)
# 🏡 The Real Estate Crystal Ball: Predicting Prices with Smarter Data

### **TL;DR**  
Grouped property features based on real estate logic, tested multiple models, and found that **feature engineering significantly improved linear regression** but had **limited impact on advanced models like CatBoost**, which ultimately performed best.  

---

## **📌 The Non-Technical Summary**  
Michael Lewis writes to **one specific person**—someone real, someone who will challenge him.  
**RP, this project is for you.**  

Most property valuation models focus on **individual features** with high correlation, but real estate professionals think about **feature interactions**—location, condition, and land use **together**.  This project tested whether **machine learning models could think more like an appraiser.**  

### **Key Takeaways:**  
- ✅ **Feature engineering improved traditional models** (Linear Regression RMSE ↓ **24,471 → 21,975**).  
- ✅ **Advanced models (Gradient Boosting, CatBoost) saw minimal gains**—they already capture relationships effectively.  
- ✅ **CatBoost performed best overall** (**RMSE 19,875**).  

While this study focused on residential data, the approach applies to commercial real estate.  

Oh, and one more thing: **this is real-world, ugly data.** It’s full of missing values, inconsistencies, and weird edge cases—just like any dataset you’d find in commercial real estate. Cleaning it up and making it usable was half the battle.  

---

## **🎯 Project Approach**  

### **🔹 Feature Engineering with Real Estate Logic**  
Instead of just using high-correlation variables, features were **grouped logically**:  
- **Interior:** Square footage, HVAC, bedrooms  
- **Exterior & Land:** Lot size, frontage, porch, garage  
- **Basement & Storage:** Finished space, bathrooms  
- **Sales & Transactions:** Sale type, year  

This structure helped traditional models **see relationships more clearly**—like combining all porch-related features into a single "outdoor space" metric.  

---

## **📌 Why Invest in EDA?**  

I chose this project as an opportunity to put **domain knowledge to work**. While feature engineering did not always improve performance, I learned in a previous project that **time spent on EDA—truly understanding the variables and their context to the desired outcome—is an investment that pays off significantly in the long run**.  

Many data science workflows **move through EDA quickly**, focusing on cleaning and preprocessing before diving into modeling. However, a deeper exploration of the dataset not only helps with **feature selection and engineering** but also improves **interpretability, error analysis, and long-term model stability**.  

Even though this approach didn’t drastically shift model performance, it strengthened my ability to **spot meaningful relationships between property features** and recognize **which insights were useful versus which were noise**.  

---

## **📊 Visualizing Key Relationships**  

EDA is not just about cleaning data—it’s about understanding how features interact with the target variable. To explore these relationships, I included key visualizations:  

- **Sale Price Distribution** – Reveals the skewness of the target variable, highlighting the need for potential transformations.  
- **Sale Price vs. Total Rooms** – Examines whether more rooms consistently lead to higher prices or if other factors play a role.  
- **Sale Price vs. GrLivArea (Above-Ground Living Area)** – A crucial check, as square footage is often one of the strongest predictors of home value.  
- **Sale Price by Decade** – Investigates whether older homes follow different pricing trends, capturing potential temporal effects.  

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/1a77c594-da71-46cb-b89e-4fe876ddd06c" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/c109c5dc-5989-4e7f-b6c3-08c77cd22dab" width="400"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/b22da2a0-6492-47e0-bcd6-a4940090f86a" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/09c1ea12-1349-452d-a032-cd4b95909a4b" width="400"></td>
  </tr>
</table>

These charts help **validate assumptions**, **identify outliers**, and **inform feature engineering decisions**, ensuring that the model is built on meaningful, structured insights rather than just raw correlations.  

---

## **📊 Model Performance**  

- **Feature engineering improved linear regression the most.**  
- **Tree-based models saw marginal improvements.**  
- **CatBoost outperformed all others.**  

| Model | RMSE Before | RMSE After  |
|---------------|------------|------------|
| **Linear Regression** | 24,471 | 21,975 |
| **Decision Tree**  | 37,033 | 36,282 |
| **Gradient Boosting** | 21,995 | 20,250 |
| **CatBoost**   | 22,427 | **19,875** |


<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/03f2ef38-badd-461a-8c5d-5bfc421e35aa" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/29b453b5-6a2d-47ae-afc3-da64e2ade0c3" width="400"></td>
  </tr>
</table>
---

## **🔎 Findings**  
1️⃣ **Feature engineering helps simpler models but adds little value to Gradient Boosting models.**  

2️⃣ **Grouping Related Features Improved Interpretability but Had Mixed Impact on Performance**  

3️⃣ **Future work:** Model stacking, hierarchical models, and better neighborhood-based composite features.  

### **🏆 Kaggle Score: 0.12897** (Top 20%)  
CatBoost delivered the strongest performance, underscoring the power of boosting algorithms in real estate valuation. The **Kaggle score of 0.12897 (Top 20%)** is excellent, especially considering that **minimizing RMSE was not the primary goal**. This project focused on comparing feature engineering methodologies rather than hyperparameter tuning, reinforcing that advanced models like CatBoost inherently capture complex relationships with minimal manual intervention.

---

## **🏁 Conclusion**  
**Domain-driven feature engineering significantly benefits traditional models, but advanced tree-based models already capture these relationships.**  
