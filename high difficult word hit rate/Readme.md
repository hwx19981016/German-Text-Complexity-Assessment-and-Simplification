difficult_words.json – List of high-difficulty words

ensemble_scores_gbert.csv – Test output result for predicting German text complexity using TextComplexityDE. Not used in the paper, but demonstrates the proper use of the dataset.

Final_ratings_table.json – Test output result for predicting German text complexity using TextComplexityDE. Not used in the paper, but demonstrates the proper use of the dataset.

hygi.py – Used for comprehensive testing (including difficult word hit rate) on `raw_output`, `output1`, and `VIBoutput_D6` (this is used). These three outputs are from another project, placed here for computation due to server architecture issues. The `test.py` in the other project is basically the same as this one, except this version includes the difficult word hit rate.

nn.ipynb – Test workflow using Captum to verify the usability of the structure for identifying "the words most related to the final result".

simp.ipynb – Workflow that uses Captum to finally calculate the difficult word hit rate and performs preliminary verification.

TextComplexityDE19.xlsx – Dataset

VIBoutput_D.json, VIBoutput_D1.json, VIBoutput_D... – These are output results from the previous project.


