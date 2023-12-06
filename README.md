# FeatureEval_MER

This repository contains the code necessary to reproduce the outcomes presented in the submission to CMMR2023 entitled "Music Emotions in Solo Piano: Bridging the Gap Between Human Perception and Machine Learning". In order to carry out the experiments, please follow the instructions below:

1. To generate the Gold Standard and reproduce the results from Section 3 (Gold Standard Assessment), go to the folder User_Study:<br> 
execute GoldStandard.py (will extract the Gold Standard from the annotations)<br> 
execute User_Study.Rmd (will perform the Gold Standard Assessment) 

2. To reproduce the experiments used to answer RQ1, go to the folder Features_Assessment:<br> 
execute Features_Correlation.Rmd (will perform the Correlation Analysis)<br> 
execute Multiple_Regression.Rmd (will perform the Multiple Regression)<br> 
execute ConfMat_Perception.py (will compute the confusion matrix for perception interpreted as NMDS in "Perception vs Classification")<br> 
execute Classification.py (will perform the ML experiments whose results are interpreted as NMDS in "Perception vs Classification")

3. To reproduce the experiments used to answer RQ2, go to the folder ML_Task:<br> 
execute EMOPIA_4Q.py (will perform the ML experiments whose results are displayed in Table 3)

If you find the content of this repository useful, you might consider giving us a citation:

E. Parada-Cabaleiro, A. Batliner, M. Schmitt, B. Schuller, & M. Schedl (2023), Music Emotions in Solo Piano: Bridging the Gap Between Human Perception and Machine Learning, in Proc. of the Int. Symposium on Computer Music Multidisciplinary Research (CMMR), Tokyo, Japan, pp. 587-598.
