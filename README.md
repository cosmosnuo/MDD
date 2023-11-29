# MDD
MDD: Process Drift Detection in Event Logs Integrating Multiple Perspectives

Python packages required
	lxml (http://lxml.de/)
	numpy (http://www.numpy.org/)
	networkx (https://networkx.org/)
	scipy (https://scipy.org/)
	torch (https://pytorch.org/)
	sklearn(https://scikit-learn.org/)


How to use **************************************************

Command line usage:

	MDD.py [-w value] [-j value] [-n value] [-f value] 
	options:
    		-w window size, integer, default value is 200
    		-j jump distance, integer, default value is 3
    		-n change number, integer, default value is 3
      		-f log, char, default value is cm5k.mxml

Examples:

	MDD.py -w 200 -j 3 -n 3 -f cm5k.mxml

 	
Note: 
	if you have 'Error reading file ', please using absolute path of log file.
	

A reference for users **************************************************

	If you have any quetions for this code, you can email:XXXX.
	We would also be happy to discuss the topic of drifts detection with you.
