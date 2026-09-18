# Agentic_Data_pipeline_With_ML_setup
Agentic API ingestion in Mage, followed by Data orchestration, a landing zone in Databricks, and automated ML with Claude Code Reviews.

Looking at economic data as it relates to unemployment (historical data, time series, early 2000s-2026).

Focus of the project is not the code itself, but the architecture design, workflow, and understanding data orchestration and batch processing.
Many enterprise systems have different tools/software for different sections of data (one for each buisness department, usually thousands if not millions of rows across a corporation) so the idea of data orchestration becomes more impactful as businesses grow.

This is an exploratory project with a focus on data orchestration + agentic workflows/code reviews. 

# Diagram
<img width="5792" height="2235" alt="Agentic_Data_Eng_ML_pipeline" src="https://github.com/user-attachments/assets/ce5e5557-84e8-4c1a-aa72-a5c8322007c8" />

 # Tech Stack

- FRED Economic Data (St. Louis) 

- Mage (Agentic Data ingestion, Orchestration and exporting. Uses Tranformer functions, data loaders, etc.) (Human in the loop)

- Medallion architecture in Mage (Human in the loop)

- Databricks (landing zone for silver and gold analytics)

- Databricks Genie exporatory data analysis 

- Model selection/architeture choices (Human in the loop)

- Cyclical Claude Code review (Opus 5) for Databricks MLFlow experiment (context engineering)