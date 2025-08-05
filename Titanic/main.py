import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("titanic.csv")
print(df.head())
df["Age"] = df["Age"].fillna(df["Age"].mean())

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df = df.drop(["PassengerId","Name","Ticket","Fare","Cabin","Embarked","Parch","SibSp"],axis=1)


#######

X = df.drop("Survived",axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
      "LogisticRegression": (
            LogisticRegression(class_weight='balanced'),
            {"model__C": [0.000001,0.000001,0.000001], "model__solver":['lbfgs','liblinear']}
      ),
      
      "RandomForestClassifier": (
            RandomForestClassifier(),
            {'model__n_estimators':[50,100,200], 'model__max_depth':[3,5,7]}
      ),
      
      "DecisionTreeClassifier": (
            DecisionTreeClassifier(),
            {'model__max_depth':[3,5,7]}
      ),
      "GaussianNB": (GaussianNB(),{}),
      
      "KNeighborsClassifier":(
            KNeighborsClassifier(),
            {'model__n_neighbors':[3,5,7]})
}

numerical_features = ['Pclass','Age','FamilySize']
categorical_features = ['Sex']
preprocessor = ColumnTransformer([
      ('num',StandardScaler(),numerical_features),
      ('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
])

print(df.head())
for name,(model,param_grid) in models.items():
      pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
      ])
      
      grid = GridSearchCV(pipe, param_grid, cv=5)
      grid.fit(X_train,y_train)
      
      print(f"\nModel: {name}")
      print("Best Params:", grid.best_params_)
      print("Train Score:", grid.best_score_)
      
      # !! In this dataset, the most powerful feature is gender.!!