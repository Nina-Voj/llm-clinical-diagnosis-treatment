# Large Language Models for Clinical Diagnosis and Treatment Recommendation
## Overview
This repository contains the code, data processing pipelines, and evaluation frameworks for a Bachelor’s project. The study investigates the efficiency of various LLMs in interpreting patient symptoms and suggesting potential diagnoses. This research utilizes the MIMIC-IV database to provide real-world context for symptom interpretation and diagnostic reasoning.
## Data & Compliance
**Source:** All clinical data used in this study is sourced from the MIMIC-IV clinical database.  
**Access:** Access to this data was granted through PhysioNet after fullfillment of the appropriate requirements.  
**Privacy:** The files from the complete version of MIMIC-IV are a restricted-access resource, no raw MIMIC-IV data is hosted in this repository. Scripts are provided to process the data locally once the user has obtained their own authorized access.
## Key Features
**MIMIC-IV Integration:** Pipelines for extracting relevent patient informtation and clinical notes from the hosp and icu modules.  
**Data processing:** Python-based scripts that transform raw patient records into strucutred datasets  and structured prompts suitable for Large Language Models.
