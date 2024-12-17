# **Personalised-Marketing-Campaign Generator**

## **Overview**

The AI-based, Personalised Marketing Campaign Generator is designed as a tool to help marketers segregate customers based on their demographics and behaviours, and then use LLMs to create Personalised Campaigns for them. The tool can also be leveraged to create targetted emails for the customers, based on the generated campaign.

## **Details**

### **EDA and Feature Engineering**
- The Dataset consists customer purchase data, with features determining the customers demographics and the bought product.
- The Dataset is cleaned and visualised in the EDA notebook.
- In the Feature Engineering notebook, new features are created for determining the behaviour of each customer.
### **ML Modelling**
- The pipeline notebook, scales and transforms the data to prepare it for modelling.
- The Modelling notebook is where the ML experimentaions was done, to find it best model for the task.
- In the Customer-Segmentation notebook, the optimized model was trainined and evaluated and the characteristics of each cluster was determined.
### **LLM Integration**
- Once the model was capable of segmenting new data into their specific category, an LLM was integrated to the application.
- The customer information and the cluster characteristics is passed to the LLM to generate the campaign strategy and the email.
### **UI**
- Finally, a streamlit UI was built for the application, where a marketer can drop the user details to generate marketing campaign and emails.
