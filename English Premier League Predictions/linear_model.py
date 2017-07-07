import pandas as pd
import tensorflow as tf
import numpy as np
import numpy as np

EPL_TRAINING = "england_training.csv"
EPL_TESTING= "england_test.csv"
EPL_PREDICT= "englandpredict.csv"
EPL_VALIDATION= "england_validation.csv"

COLUMNS = ["season","home","visitor","hgoal","vgoal",
            "divison","tier","totgoal","goaldif","result"]
df_train = pd.read_csv(tf.gfile.Open(EPL_TRAINING), names=COLUMNS,
                        skipinitialspace=True, engine="python")
df_test = pd.read_csv(tf.gfile.Open(EPL_TESTING), names=COLUMNS,
                        skipinitialspace=True, engine="python")
df_predict = pd.read_csv(tf.gfile.Open(EPL_PREDICT), names=COLUMNS,
                        skipinitialspace=True, engine="python")
df_validation = pd.read_csv(tf.gfile.Open(EPL_VALIDATION), names=COLUMNS,
                        skipinitialspace=True, engine="python")

LABEL_COLUMN = "label"
df_train[LABEL_COLUMN] = df_train["result"].astype(int)
df_test[LABEL_COLUMN] = df_test["result"].astype(int)
df_predict[LABEL_COLUMN] = df_predict["result"].astype(int)
df_validation[LABEL_COLUMN] = df_validation["result"].astype(int)

CONTINUOUS_COLUMNS = ["hgoal","vgoal","totgoal","goaldif","result"]
CATEGORICAL_COLUMNS = ["divison", "tier", "home", "visitor", "season"]
'''Creates My Dictionary Mappings'''

def input_fn(df):
    # Store the values of my CONTINUOUS_COLUMNS in a constant tensorflow
    continuous_cols = {k: tf.constant(df[k].values)
                        for k in CONTINUOUS_COLUMNS}
    # Create a dictionary mapping from each  categorical feature
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].astype(str).values,
        dense_shape=[df[k].size, 1])
                        for k in CATEGORICAL_COLUMNS}
    # Merg them together
    feature_cols = dict(continuous_cols.items() + categorical_cols.items())
    # Make the super cool tensorflow
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label

# This returns the input stuff for our really cool tensor contruction
def train_input_fn():
    return input_fn(df_train)
def eval_input_fn():
    return input_fn(df_test)
def predict_input_fn():
    return input_fn(df_predict)
def valid_input_fn():
    return input_fn(df_validation)

# Since we're too cool lets to assign all the teams numbers let's use a hash bucket
# and we should for the other categorical data
home = tf.contrib.layers.sparse_column_with_hash_bucket("home", hash_bucket_size=100)
divison = tf.contrib.layers.sparse_column_with_hash_bucket("divison", hash_bucket_size=100)
tier = tf.contrib.layers.sparse_column_with_hash_bucket("tier", hash_bucket_size=100)
season = tf.contrib.layers.sparse_column_with_hash_bucket("season", hash_bucket_size=100)
visitor = tf.contrib.layers.sparse_column_with_hash_bucket("visitor", hash_bucket_size=100)

# Continuous Feature Power
hgoal = tf.contrib.layers.real_valued_column("hgoal")
vgoal = tf.contrib.layers.real_valued_column("vgoal")
totgoal = tf.contrib.layers.real_valued_column("totgoal")
goaldif = tf.contrib.layers.real_valued_column("goaldif")
result = tf.contrib.layers.real_valued_column("result")
# Since we really don't have correlations we don't
# need bucketization or a cross reature column

'''Build the Model. It automatically learns a bias term'''
model_dir = "/tmp/EPL_POWER"
# L1 tends to make the model stay at 0
# L2 makes the model's weights closer to zero
m = tf.contrib.learn.LinearClassifier(feature_columns=[
    season, home, visitor, divison, tier, hgoal, vgoal, totgoal, goaldif,result],
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.01,
        l1_regularization_strength=0.5,
        l2_regularization_strength=0.5),
    model_dir=model_dir)
# training
m.fit(input_fn=train_input_fn, steps=200)
# evaluate
results = m.evaluate(input_fn=valid_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

#pred_proba = m.predict_proba(input_fn=lambda: input_fn(df_predict))

#for x in pred_proba:
#    print(x)
