# DMwrapper: Weka Attribute Selection Tool

This repository provides a Python-based wrapper for **Weka** that automates the process of attribute selection to optimize classification performance. It utilizes a greedy search strategy to identify the most influential variables in a dataset for a **Random Forest** model.

## Features

* **Automated Greedy Selection**: Implements an iterative forward-selection-like approach to find which attributes yield the highest accuracy (asmatze tasa) when evaluated via cross-validation.
* **Weka Integration**: Leverages the `python-weka-wrapper` to access Weka's core functionalities, including ARFF loading, filtering, and classification.
* **Random Forest Classifier**: Uses a pre-configured Random Forest model (100 trees, 100 iterations) as the base evaluator for attribute subsets.
* **Cross-Validation**: Ensures robust performance metrics by using 10-fold cross-validation during the selection process.
* **Visual Analysis**: Generates a performance plot using `matplotlib` to visualize how accuracy improves as more significant variables are identified.

## Requirements

To run this tool, you need Python installed along with the following dependencies:
* **Java Runtime Environment (JRE)**: Required for the Weka JVM.
* **`python-weka-wrapper3`**: The interface between Python and Weka.
* **`matplotlib`**: Used for plotting the final results.

## Usage

1.  **Start the Program**: Run the script using Python:
    ```bash
    python programa.py
    ```
2.  **Dataset Input**: When prompted, enter the name of the `.arff` file you wish to analyze.
3.  **Iterative Selection**: The program will begin testing different combinations of attributes. It will print the "highest variable" (Aldagai altuena) and its corresponding accuracy rate at each step.
4.  **Results**: Once the selection process stops (when adding more variables no longer improves performance), the program will list the final set of selected attributes and the total accuracy achieved.
5.  **Visualization**: A graph will be displayed showing the accuracy progression across the selected attributes.

## Core Logic Overview

The script (`programa.py`) follows these steps:
* **JVM Startup**: Initializes the Weka environment.
* **Data Loading**: Loads the specified ARFF file and sets the class attribute to the last position.
* **Filtering**: Dynamically applies a `Remove` filter to isolate specific attribute subsets for testing.
* **Evaluation**: Builds a `RandomForest` classifier and evaluates it using `Evaluation.crossvalidate_model`.
* **Greedy Loop**: Continues adding the best-performing attribute to the selection list until no further improvement in the "percent correct" (asma\_tasa) is observed.
