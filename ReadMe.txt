#Execution steps for User Interface with HTML,CSS,bootstrap and JS
python ./api.py


#Execution steps for User interface with Dialogflow 

#step 1:  Dialog flow UI
python ./DialogFlow_Integration/webpage.py 


#step 2:  Rest API
python ./DialogFlow_Integration/df_api.py 


#pep8 python coding standards
flake8_nb api.py
flaske8_nb ./DialogFlow_Integration/webpage.py
flaske8_nb ./DialogFlow_Integration/df_api.py 
