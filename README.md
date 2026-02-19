1. The code was successfully run on an Arch Linux environment
2. The version of python used was 3.11.14. The requirements are known to not be possible to satisfy with python 3.13
3. The `requirements.txt` file lists all the libraries that are needed to be installed.
4. Java should be installed on the system since that is required for the admg to pag conversion. The latest version of Java (Java 25 JDK OpenJDK) gave warnings but ran correctly. Java 21 OpenJDK did not produce those warnings.
5. The implementation of the DCD algorithm was obtained from `https://gitlab.com/rbhatta8/dcd` and slightly modified to fit the api required for the experiments.
6. The DCD algorithm also had a non-functioning java integration with the tetrad library since the API of the tetrad library has evolved since they implemented it. All changes made in the dcd code is commented with `danish mod` to explicitly highlight the changes made.
7. There were two main files for running experiments: `relcadilac/experiments.py` and `cma_es/experiments.py`. The former was the old experimental setup which often involved repetitive code, and was discarded. The latter is the new intended approach for use.
8. The starting file to read / understand the algorithm implementation for Relcadilac is `relcadilac/relcadilac.py`.
9. The starting file to read / understand the algorithm implementation for the CMA-ES version of the framework is `cma_es/cma_es.py`.
10. The primary file for generating sample graphs and sampling data from them is the `relcadilac/data_generator.py`. The primary function intended for use is the `GraphGenerator.get_admg` function in that file.
11. The modified implementation of the ananke-causal library's RICF algorithm is given in the file `relcadilac/optim_linear_gaussian_sem.py` and is standalone in that it does not call any other functions in other local files - only libraries.
12. The `relcadilac/metrics.py` file is used to obtain metrics on comparisons between ground truth pag and admg graphs.
13. The intended way to run the algorithms on the sachs dataset is also through the `cma_es/experiments.py` file (see the `run_sachs_dataset` function in that file.
14. The intended way to run the `cma_es/experiments.py` file is to update the parameters in the constructor of the `Experiments` class and the parameters in the following functions:
    1. set_graph_generation_params
    2. set_post_prediction_params
    3. set_cmaes_params
    4. set_relcadilac_params
    5. set_dcd_params
15. The file level `run_*` functions in the `cma_es/experiments.py` file are illustrative on how I was running the experiments and controlling the parameters.
16. If the CMA-ES or Relcadilac code is run directly from their respective files, it is likely to run poorly since they are multiprocess algorithms, and these clash with the normal behaviour of numpy (at least on Arch Linux). Thus, those if those files are to be run directly they must be modified to have the following at the very top of the file (before loading numpy):
    ```
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    ```
17. The results of the experiments / runs from `cma_es/experiments.py` are added to the `runs/runs.csv` file, the diagrams (pdf plots of the ADMG graphs) are added to the `diagrams` folder, a pickle file with the run number will be saved to the `runs` folder (which contains a dictionary of objects that could not be effectively saved to the csv file - like lists of rewards, or the dataset, etc - this essentially helps guarantee reproducibility of results), and if the CMA-ES algorithm was run, it will produce a folder for each run for the defualt outputs of the library implementation of the algorithm.
18. It is highly recommended to use a python virtual environment since the ananke-causal python library is a little finicky in terms of compatibility with other libraries.
19. The `cma_es/experiments.py` run does not create the `runs/runs.csv` file or add column names in it, so a starter `runs.csv` file with the correct order of column names is provided.
20. All files are intended to be run from the root folder as: `python -m FOLDER_NAME.FILE_NAME` where the FILE_NAME has no extension (of .py)
21. A `run_demo` function is provided in the `cma_es/experiments.py` file to check if all the algorithms are working as expected. It quickly runs with small values and adds 4 entries (one for each algorithm) to `runs/runs.csv`.
22. The runs are intended to continue even in case of errors - they print the error and move on, to ensure that one failure does not impact other in case of large experiments.
23. The current state of the repository is as after the `run_demo` function was run
24. The code is also available on the github repository `https://github.com/TheDansmod/qmul-msc-ai-code-master-thesis`
