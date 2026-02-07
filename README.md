# 📊 SaaS Churn, Unit Economics & Customer Segmentation Dashboard

**Project: RavenStack (Synthetic SaaS Dataset)**

---

## Project Background

This project focuses on analyzing **customer churn behavior, unit economics (LTV/CAC), and customer segmentation** for a B2B SaaS product named **RavenStack**, an AI-driven team productivity platform.

The objective of the analysis is to understand:

* why customers churn,
* which acquisition channels and customer segments are high risk,
* how unit economics have evolved over time,
* and how customer value differs across regions, plans, and industries.

The dataset used in this project is **fully synthetic but statistically realistic**, generated to simulate real-world SaaS business behavior. It captures the complete customer lifecycle, including signups, subscriptions, feature usage, support interactions, upgrades/downgrades, and churn events.

the dashboard is at : https://jenifermariajoseph-churn-customer-segmentation-dashboard-donbvu.streamlit.app/

---

## Dataset Overview

The dataset represents a multi-table SaaS environment with referential integrity:

* **Accounts**: customer demographics, plan at signup, churn flag
* **Subscriptions**: billing details, plan tier, MRR, auto-renew status
* **Feature Usage**: granular product engagement metrics
* **Support Tickets**: customer support load and satisfaction
* **Churn Events**: churn dates and reason codes

All data is synthetic and created for analytical demonstration purposes only.

---

## Key Areas of Analysis

Insights and observations are developed across the following areas:

1. **Churn Trend Analysis**
2. **Churn Probability by Customer Segment**
3. **Churn Reason Distribution**
4. **Unit Economics (LTV/CAC)**
5. **Plan Mix & Revenue Drivers**
6. **Customer Segmentation using Clustering**
7. **Predictive Churn Modeling**

---

## Executive Summary

### Overview of Findings

* Monthly churn shows a **clear upward trend**, indicating increasing customer attrition as the product scales.
* Churn probability is **significantly higher** for customers acquired via **partner, organic, and event-based channels**, compared to paid ads.
* **Basic plan customers** show consistently higher churn risk than Pro and Enterprise users.
* Customers with **low product usage and high support dependency** are the most likely to churn.
* Unit economics reveal that **LTV/CAC ratios have declined over time**, approaching break-even levels in later months.
* Enterprise and Pro plans contribute disproportionately to LTV despite representing a smaller share of total accounts.
* Customer clustering reveals four distinct personas with clearly different value and churn characteristics.

---

## Churn Trend Analysis
<img width="437" height="387" alt="Screenshot 2026-02-07 at 2 41 45 PM" src="https://github.com/user-attachments/assets/bbc5a3c6-90c9-4b3c-9ccd-983f97138ee7" />


Monthly churn counts increase steadily over time, with sharper rises observed in later periods.

<img width="1436" height="764" alt="Screenshot 2026-02-07 at 4 16 43 PM" src="https://github.com/user-attachments/assets/ead76dab-ba9d-436d-9922-b1bce1fd0eb5" />
this might be primarily due to increase in  pricing seen in later months leading customers to churn 

---

## Churn Probability Analysis

Using a supervised machine learning model (Logistic Regression with class balancing), churn probabilities were predicted at the **customer level** and then aggregated across key dimensions.

### Churn Probability by Referral Source

* **Partner and Organic channels** exhibit the highest average churn probability.
* **Paid Ads** show comparatively lower churn risk, indicating higher acquisition quality.

### Churn Probability by Country

* Churn risk varies meaningfully across regions.
* Certain countries show consistently higher churn probabilities, highlighting the need for localized retention strategies.

### Churn Probability by Industry

* Industries with complex workflows and higher collaboration needs (e.g., DevTools, FinTech) show lower churn.
* More price-sensitive industries exhibit higher churn risk.

---



## Unit Economics (LTV / CAC)

### LTV/CAC Ratio Over Time
<img width="446" height="367" alt="Screenshot 2026-02-07 at 4 34 48 PM" src="https://github.com/user-attachments/assets/0fe67fba-3c06-485c-921f-e4bfd5fcf1ef" />

* Early cohorts show healthy LTV/CAC ratios.
* Ratios decline over time, approaching the break-even threshold.
* This trend suggests rising acquisition costs and/or declining lifetime value.

### Plan Mix Distribution

* We can see that number of pro customers increase towards the end of the year.
* however since we can see that the churn still increases towards the end of the year and we may need to look into that.

### Average LTV Analysis

* LTV varies significantly by country and industry.
* Certain regions deliver higher revenue per customer despite lower volumes.

---

## Customer Segmentation Analysis
<img width="1281" height="527" alt="Screenshot 2026-02-07 at 5 41 38 PM" src="https://github.com/user-attachments/assets/b46f5fbd-e24a-4c5b-b444-10ce31a37af7" />

Using K-Means clustering, customers were segmented based on:

* MRR
* Seat count
* Product usage
* Support ticket volume

### Identified Customer Segments

1. **High Value – Power Users**

   * High usage, high MRR, low churn risk

2. **Low Value – High Support**

   * High support dependency, elevated churn risk

3. **Low Value – Low Usage**

   * Poor engagement, highest churn risk

4. **Mid Value – Growing Accounts**

   * Moderate usage and revenue, strong upsell potential

Segment distribution varies significantly by country, industry, and plan tier.

---

## Business Recommendations

### 1️⃣ Improve Low-Usage Customer Activation

* Target onboarding and in-app guidance for low-usage accounts.
* Focus on the first 30–60 days post-signup.

### 2️⃣ Strengthen Retention for Basic Plans

* Introduce feature gating, upgrade nudges, or usage-based incentives.
* Evaluate pricing and perceived value of Basic plans.

### 3️⃣ Optimize Acquisition Channels

* Re-evaluate partner and organic acquisition strategies.
* Focus spend on channels with lower churn probability.

### 4️⃣ Reduce Support-Driven Churn

* Identify recurring support issues.
* Invest in self-serve documentation and product stability.

### 5️⃣ Focus on High-Value Segments

* Prioritize Enterprise and Power Users for retention and expansion.
* Use segmentation insights for personalized outreach.

---

## Assumptions & Caveats

* All data is **synthetic** and generated for analytical purposes.
* Revenue, churn, and usage metrics are simulated and not real transactions.
* Insights are **directional**, intended to demonstrate analytical thinking rather than real-world performance.
* The project focuses on methodology, modeling, and storytelling.

---

## Tools & Technologies Used

* Python (Pandas, NumPy, Scikit-learn)
* Logistic Regression & K-Means Clustering
* Data Visualization (Tableau / Matplotlib / Plotly)
* Jupyter Notebook

---




