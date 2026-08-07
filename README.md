\# Zepto AI ML Capstone Project



\## Overview



This repository contains an end-to-end AI/ML engineering capstone project with three connected modules:



1\. Data Engineering Pipeline

2\. Analytics and Machine Learning Pipeline

3\. GenAI Customer Support Assistant



\## Project Structure

zepto-capstone/



├── data\_pipeline/

│   └── Data engineering pipeline



├── analytics/

│   └── Analytics and machine learning workflow



├── support\_assistant/

│   └── Document-grounded customer support assistant



└── requirements.txt





\## Setup



Install dependencies:



pip install -r requirements.txt





\## Modules



\### Data Pipeline



Handles data collection, cleaning, storage, and querying.



Folder:



data\_pipeline/





\### Analytics Pipeline



Performs data analysis, visualization, model training, and evaluation.



Folder:



analytics/





\### Support Assistant



A Streamlit assistant that answers questions using Zepto policy documents.



Folder:



support\_assistant/





Run the assistant:



cd support\_assistant



python -m streamlit run app.py





\## Design Decisions



\- Data pipeline creates clean structured data.

\- Analytics pipeline follows the machine learning lifecycle.

\- Support assistant uses policy documents to provide grounded answers.

\- All modules are maintained in one repository.

