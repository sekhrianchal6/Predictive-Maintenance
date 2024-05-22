/************** Model Development *********************/
- Train/Validation Split:
	- 70/30 split
	- Splitting the dataset based on the Engine feature
	- Ex: Set of samples(rows) corresponding to Engines 1 to 70 are used for training the model
		  Remaining set of samples(rows) corresponding to Engines 71 to 100 are used for testing the model
	- Using groupshufflesplit() helps to acheive this, please refer to template.ipynb
	
- XTrain: (Features)
	- Following features from original dataset are transformed (exponential weighted mean) and scaled (Standard Scaling) except Cycles
	- Cycles, Sensor5, Sensor6, Sensor7, Sensor10, Sensor11, Sensor12, Sensor14, Sensor15, Sensor16, Sensor17, Sensor18, Sensor20, Sensor23, Sensor24
	- Path: Predictive-Maintenance\Datasets\ForModelDev\XTrain.csv
	
- YTrain: (Target)
	- Remaining Cycles
	- Path: Predictive-Maintenance\Datasets\ForModelDev\YTrain.csv