# ABM_assignment

Agent-based model of household recycling behavior using Mesa.

Files and folders description:

model:
    agents.py  
    → Defines the Household and Bin agent classes

    core.py  
    → Implements the RecyclingModel class

analysis:
    basic_run.py  
    → Runs a basic simulation of the model
    → Here you can find a description of every parameter

    avalanches_sample.py  
    → Analyzes avalanche events in the simulation 

    multiple_outputs_analysis.py
    → Run avalanches analysis for different output measures

    sensitivity_analysis.py  
    → Run sensitivity analysis for the model

data:
    → Contains results for sensitivity analysis for replotting without having to run the SA again

figures:
    → Contains every figure and animation generated through any of the code

drafts_and_notebooks:
    → Contains the notebooks used to test the model, generate animations and investigate emergence