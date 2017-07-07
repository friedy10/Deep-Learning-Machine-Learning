import tensorflow as tf
import tempfile
import pandas as pd

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

# Let's do some word embedding
deep_columns = [tf.contrib.layers.embedding_column(season, dimension=8),
    tf.contrib.layers.embedding_column(home, dimension=8),
    tf.contrib.layers.embedding_column(divison, dimension=8),
    tf.contrib.layers.embedding_column(tier, dimension=8),
    tf.contrib.layers.embedding_column(visitor, dimension=8),
    hgoal,vgoal,totgoal,goaldif, result]

model_dir = "/tmp/DNN_ENGLAND"
m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir = model_dir,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])

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

df_train["divison"] = df_train["divison"].astype(str)
df_test["divison"] = df_test["divison"].astype(str)
df_predict["divison"] = df_predict["divison"].astype(str)
df_validation["divison"] = df_validation["divison"].astype(str)

df_train["tier"] = df_train["tier"].astype(str)
df_test["tier"] = df_test["divison"].astype(str)
df_predict["tier"] = df_predict["tier"].astype(str)
df_validation["tier"] = df_validation["tier"].astype(str)

df_train["season"] = df_train["season"].astype(str)
df_test["season"] = df_test["season"].astype(str)
df_predict["season"] = df_predict["season"].astype(str)
df_validation["season"] = df_validation["season"].astype(str)

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
        values=df[k].values,
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


m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=valid_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

pred_proba = m.predict_classes(input_fn=lambda: input_fn(df_predict))

for x in pred_proba:
    print(x)
