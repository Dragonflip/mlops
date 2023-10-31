from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import os

prefix = 'sagemaker-featurestore-introduction'
role = os.environ.get('aws_role')

sagemaker_session = Session()
region = sagemaker_session.boto_region_name
s3_bucket_name = sagemaker_session.default_bucket()

feature_group_name = 'test-feature-group-30-17-09-25'


feature_group = FeatureGroup(
    name=feature_group_name, sagemaker_session=sagemaker_session
)

query = feature_group.athena_query()
table_name = query.table_name

query_string = ('SELECT * FROM "%s"' % table_name)

query.run(query_string=query_string,
          output_location=f's3://{s3_bucket_name}/{prefix}/query_results/')
query.wait()
dataset = query.as_dataframe()

X=dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y=dataset[['target_name']]
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.5, shuffle=True,random_state=100)

model=SVC(C=1, kernel='rbf', tol=0.001)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
