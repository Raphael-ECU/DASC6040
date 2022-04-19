# DASC6040

4/19/2022
Raphael DuSablon

This repo contains a reworking of the codebase provided by Silburt et al in the following repository: https://github.com/bloomberg/fast-noise-aware-topic-clustering

This codebase includes the addition of an approximation of the dpmeans algorithm, as well as a script to run both dpmeans and FANATIC on subsections of the data set.

Consult the authors' ReadMe in their repo for initialization of this codebase, including their pip install instructions for setting up the enviroment and their directions for downloading the data set. Once that is complete, run the main_driver.py script to process the data.

Of Note: in addition to the default lambda hyper parameter settings, which can be statically assigned in the arguments.py module, this codebase includes the option to set a dynamic lambda value. The current pseudodpmeans.clustering.dpmeans.py module has this option enabled at line 154. To disable, replicate the author's original code for this line, as can be seen at fanatic.clustering.fanatic.py line 428.

This codebase has been run in three different iterations with different lambda values, 0.324, 0.648, and dynamic. The outputs of those runs have been uploaded, minus the json files to conserve space.
