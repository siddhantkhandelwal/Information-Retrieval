# Assignment 2

## Group 16

    - Siddhant Khandelwal (2017A7PS0127P)
    - Aryan Mehra (2017A7PS0077P)

## Setup

- Create the environment
  
  - Modify ```prefix``` in environment file to the location where you wish to install the environment
  
  ```bash
  conda env create -f environment.yml
  conda activate ir
  ```

- Launch the jupyter notebook
  
    ```bash
    jupyter notebook
    ```

## Innovations

- Uncomment the following line of code in function `scoring` to activate Heuristic 2
  ```bash
  scores = title_weighting(scores, corrected_query, doc_titles)
  ```
- Heuristic 1 has been incorporated in the natural flow of the code itself, thus, giving user the choice to select Yes/No in order to activate the heuristic at run time
- Spell check can be activated and deactivated in function `process_query_vector` 
  ```bash
  query = spell_correct(query)
  ```