import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


file_path = 'IMDb Movies India.csv'
movies_df = pd.read_csv(file_path, encoding='latin1')


movies_df = movies_df.dropna(subset=['Rating', 'Votes'])


movies_df = movies_df.dropna(subset=['Year', 'Duration', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'])


movies_df['Year'] = movies_df['Year'].str.extract('(\d{4})').astype(int)

movies_df['Duration'] = movies_df['Duration'].str.extract('(\d+)').astype(int)

movies_df = pd.get_dummies(movies_df, columns=['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], drop_first=True)

# Features and target variable
X = movies_df.drop('Rating', axis=1)
y = movies_df['Rating']


X = X.apply(pd.to_numeric, errors='coerce')

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Standardize the data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
