# Study the reliability of LLMs on Data Science Tasks

----------------------------------------------------

## Our Experimental setting 

### Change on the settings

- Fixing the prompt and the other details, use (i) model = "gpt-4o", (ii) model = "gpt-4o-mini", (iii) model = "4o-mini", also one option for vanilla LLM like without code interpreter can be tried (cheapest option for testing)
- Fixing the prompt and the model, use different temperature values by tweaking the "temperature = default" value, it can be like (0.2, 0.5, 1.0, 1.5) based on its original range
- Fixing the prompt and all other settings, use the prompt with programming language constraint or not such as "constraint: "Using the built-in functions and suitable R packages"' in the prompt or system instruction

## Our Task types and related questions at different levels 

- Data Undertanding and Wrangling
- Data Cleaning, Pre-processing and Transformations
- Data Summary Statistics and Interpretations
  
- Data Visualization - EDA 

- Statistical Models under supervised , unsupervised learning type of models
  - Supervised Learning 
    - Linear Regression modeling
    -  Variations of regression (Lasso or Ridge)
    - Classification models such as Logistic Regression model
    - Tree based model
    - Model comparison type questions
  - Unsupervised Learning
    - Clustering methods
    - Dimensionality Reductions (PCA)

- Statistical Hypothesis testing related questions
- Bayesian regression

### Possible format of the prompts

Question List: GAIL-DA-tasks-questions.jsonl file but will be expanded

- **id**: Unique identifier for each question.
- **question**: The description of the data analysis question.
- **concepts**: The concepts involved in the question.
- **format**: The format requirements for the output.
- **file_name**: The file name of the corresponding csv file.
- **level**: The difficulty level for each question.

Additional details like constraints: The constraints on the type of programming (using R instead of Python) or similar ones can be considered!
We will gather feedback from the audience for the **concepts** and the **level** information by asking them randomly allocated set of questions.


## How to conduct the experiments for new dataset or questions

To run multi-round experiments on a selected set of questions specific to one dataset, we provide a single file 
(*runner.py*) that can accommodate both independent questions and sequential questions. The adjustable setting is 
- model choice (default: 'gpt-4o')
- temperature (1.0)
- instruction for the AI assistant
- data-file-related features: dataname, filename, file_id, outfolder
- question-related features: selected questions (Qs), the number of rounds (Ns), and the number of already-done rounds (ks).

If all the questions are independent, try the function **multi_round_assistant**.
If the questions are sequentially related, try the alternative **sequential_question_assistant**.

After obtaining the experimental outcomes, we provide a single file (*evaluation.py*) that can help to collect results
and make evaluation. The adjustable setting is the same as above, except the instruction being no longer used. 
