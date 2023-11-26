
## What do anomaly scores actually mean? Key characteristics of algorithms' dynamics beyond accuracy
FIV, Nov 2023

Comparison of the dynamics of the scores generated by different outlier detection algorithms subjected to different types of perturbations: S-curves, accuracy, discriminant power, stability, robustness, confidence, coherence, variance.

Scripts to generate data, analysis and paper results:
*What do anomaly scores actually mean? Key characteristics of algorithms' dynamics beyond accuracy*
by F. Iglesias, H. O. Marques, A. Zimek, T. Zseby

### 1. Generate data

Run:

        python3 generate_data.py dataS plots

It creates the folder **[dataS]** and all datasets used for the experiments. It additionally creates the **[plots]** folder with some selected plots for the paper.


### 2. Run outlier detection algorithms on data

To run outlier detection algorithms on the generated data:

        python outdet.py dataS scores_minmax perf_minmax.csv minmax 1

The **[dataS]** folder must exist. If generated with *Step 1*, it contains datasets in .CSV format (with first row as header and the last column 'y' is the binary label: '1' for outliers). The script will generate the **[scores_minmax]** folder with a file per dataset containing the object-wise outlierness scores outputed by each tested algorithm. It also creates the file **perf_minmax.csv**, with a summary table with the overall performances (various metrics) of all algorithms for all datasets. The **minmax** argument selects the type of normalization applied on the outlierness scores. Argument **1** is just for skipping the first row of the datasets (i.e., the header).

For *proba*-normalization, run:

        python outdet.py dataS scores_proba perf_proba.csv gauss 1

Similarly, it will generate scores (**[scores_proba]**) and summaries (**perf_proba.csv**), but for *probability* normalization of scores with the **gauss** argument.


### 3. Extract proposed metrics for the dynamics of outlierness scoring

To extract the metrics proposed in the paper, run:

        python3 compare_scores_group.py dataS scores_minmax plots_minmax dyn_minmax.csv 1

or:

        python3 compare_scores_group.py dataS scores_proba plots_proba dyn_proba.csv 1

Again, the **[dataS]** is the folder with datasets. Scores obtained from outlier detection algorithms must be passed as inputs (**[scores_minmax]** and **[scores_proba]** respectively). Note that these are folder names. The *compare_scores_group.py* script matches the right dataset and file-with-scores thanks to the naming used. The script generates a folder with plots of *S-curves* (**dyn_minmax.csv** and **dyn_proba.csv** respectively) and files with a summary table of "dynamic_metric-dataset-algorithms" (**[scores_minmax]** and **[scores_proba]** respectively).


### 3. Extract Perini's metrics (Stability & Confidence)

To extract Perini's metrics (Stability & Confidence), run:

        python3 perini_tests.py dataS peri_stab_minmax.csv peri_conf_minmax.csv minmax 1

or, 

        python3 perini_tests.py dataS peri_stab_proba.csv peri_conf_proba.csv gauss 1

Again, the **[dataS]** is the folder with datasets. It generates CSV files with tables for the Stability (**peri_stab_minmax.csv** and **peri_stab_proba.csv**) and Confidence (**peri_conf_minmax.csv** and **peri_conf_proba.csv**) measurements.

Note that Perini's Confidence is defined element-wise. To obtain a Confidence value per solution we use the 1% quantile.

*Warning!! Processes in Step 3 can take considerable time on normal computers.*

#### - Sources and references 
Original scripts are obtained from the repositories:

- Confidence [1]: [https://github.com/Lorenzo-Perini/Confidence_AD](https://github.com/Lorenzo-Perini/Confidence_AD) 

- Stability [2]: [https://github.com/Lorenzo-Perini/StabilityRankings_AD](https://github.com/Lorenzo-Perini/StabilityRankings_AD)

[1] Perini, L., Vercruyssen, V., Davis, J.: *Quantifying the confidence of anomaly detectors in their example-wise predictions*. In: The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases. Springer Verlag (2020).

[2] Perini, L., Galvin, C., Vercruyssen, V.: *A Ranking Stability Measure for Quantifying the Robustness of Anomaly Detection Methods*. In: 2nd Workshop on Evaluation and Experimental Design in Data Mining and Machine Learning @ ECML/PKDD (2020).

### 4. Merging all dynamic indices

To merge all dynamic and accuracy indices in a single file (for accuracty we only keep ROC and AAP), run:

        python3 merge_indices.py dyn_minmax.csv perf_minmax.csv peri_stab_minmax.csv peri_conf_minmax.csv all_minmax.csv

or:
 
        python3 merge_indices.py dyn_proba.csv perf_proba.csv peri_stab_proba.csv peri_conf_proba.csv all_proba.csv


The only outputs generated here are **all_minmax.csv** (minmax case) and **all_proba.csv** (proba case). Other arguments refer to and are consistent with the files that we have defined in the previous points.


### 5. Scatter plots and tables comparing metrics

To generate the scatter plots in the paper that compare metrics and algorithms, run:

        python3 scatterplots.py all_minmax.csv plots_minmax

or: 

        python3 scatterplots.py all_proba.csv plots_proba


Additional plots will be generated in the **[plots_minmax]** (minmax case) and **[plots_proba]** (proba case).

You can also generate a table in .TEX format (**perf_table.tex**) with an overall comparison by running:

        python3 latex_table.py all_minmax.csv all_proba.csv perf_table.tex

Correlation plots are generated with:

        python3 metric_corr.py all_minmax.csv all_proba.csv

