import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
	df = pd.read_csv('ad.data', header=None)
	explanatory_variable_columns = set(df.columns.values)
	response_variable_column = df[len(df.columns.values)-1]
	explanatory_variable_columns.remove(len(df.columns.values)-1)

	y = [1 if e == 'ad.' else 0 for e in response_variable_column ]
	X = df[list(explanatory_variable_columns)]

	X.replace(to_replace=' *\?',value = -1,regex = True, inplace = True)

	X_train,X_test,y_train,y_test = train_test_split(X,y)


	pipeline = Pipeline([
		('clf',DecisionTreeClassifier(criterion='entropy'))
		])


	parameters={
		'clf_max_depth':(150,155,160),
		'clf_min_samples_split':(1,2,3),
		'clf_min_samples_leaf':(1,2,3)

	}

	grid_search = GridSearchCV(pipeline,parameters,n_jobs=1,verbose=1,scoring='f1')
	grid_search.fit(X_train,y_train)

	print "Best Score : ",grid_search.best_score_
	print "Best parameters set "
	best_parameters = grid_search.best_estimator_.get_params()

	for param_name in sorted(parameters.keys()):
		print(param_name,best_parameters[param_name])

	prediction = grid_search.predict(X_test)
	print prediction
	print classification_report(y_test,prediction)	
