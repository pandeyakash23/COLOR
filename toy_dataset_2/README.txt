Setting up the conda environment:
 - conda env create -f environment.yml
 - conda install anaconda::ipykernel
 - conda activate complor

Training the Com-PLOR model:
 - python train_model.py --device=cuda --num_epochs=350

Testing the Com-PLOR model:
 - python test_model.py

Steps to calculate monomeric contribution:
 1) First calculate the importance of features in 3D representation
	- python feature_imp_all.py

 2) Secondly calculate the monomeric contribution score
 	- python monomeric_contribution_calculation.py

 3) Evaluate the monomeric contribution score using masking technique
	- python masked_effect_calculation.py 
		* this code will ask for the input regarding the 
		  % of monomers to be unmasked in the sequence
	- the importance of monomers in the sequences is saved
	  as './data/test_importance.ipynb'

 Note: Steps 2 and 3 are carried out for test set only

 4) To visualize the monomeric contribution score:
    - run plot_monomeric_imp.ipynb
	  