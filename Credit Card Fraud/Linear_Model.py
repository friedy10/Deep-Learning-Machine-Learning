import tensorflow as tf
import pandas as pd
import urllib

# All my data is continuous
time = tf.contrib.layers.real_valued_column("Time")
v1 = tf.contrib.layers.real_valued_column("V1")
v2 = tf.contrib.layers.real_valued_column("V2")
v3 = tf.contrib.layers.real_valued_column("V3")
v4 = tf.contrib.layers.real_valued_column("V4")
v5 = tf.contrib.layers.real_valued_column("V5")
v6 = tf.contrib.layers.real_valued_column("V6")
v7 = tf.contrib.layers.real_valued_column("V7")
v8 = tf.contrib.layers.real_valued_column("V8")
v9 = tf.contrib.layers.real_valued_column("V9")
v10 = tf.contrib.layers.real_valued_column("V10")
v11 = tf.contrib.layers.real_valued_column("V11")
v12 = tf.contrib.layers.real_valued_column("V12")
v13 = tf.contrib.layers.real_valued_column("V13")
v14 = tf.contrib.layers.real_valued_column("V14")
v15 = tf.contrib.layers.real_valued_column("V15")
v16 = tf.contrib.layers.real_valued_column("V16")
v17 = tf.contrib.layers.real_valued_column("V17")
v18 = tf.contrib.layers.real_valued_column("V18")
v19 = tf.contrib.layers.real_valued_column("V19")
v20 = tf.contrib.layers.real_valued_column("V20")
v21= tf.contrib.layers.real_valued_column("V21")
v22 = tf.contrib.layers.real_valued_column("V22")
v23 = tf.contrib.layers.real_valued_column("V23")
v24 = tf.contrib.layers.real_valued_column("V24")
v25 = tf.contrib.layers.real_valued_column("V25")
v26 = tf.contrib.layers.real_valued_column("V26")
v27 = tf.contrib.layers.real_valued_column("V27")
v28 = tf.contrib.layers.real_valued_column("V28")
amount = tf.contrib.layers.real_valued_column("Amount")

train_file = 'creditcard_training.csv'
test_file = 'creditcard_testing.csv'
validation_file = 'creditcard_validation.csv'
predict_file = 'predict.csv'




model_dir = "/tmp/Credit Card Fraud/"
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
  v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
  v21, v22, v23, v24, v25, v26, v27, v28, amount],
  optimizer=tf.train.FtrlOptimizer(
    learning_rate=0.01,
    l1_regularization_strength=1.0,
    l2_regularization_strength=1.0),
  model_dir=model_dir)

LABEL_COLUMN = 'label'
COLUMNS = ["Time", "V1", "V2", "V3", "V4", "V5",
        "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13",
        "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21",
        "V22", "V23", "V24", "V25", "V26", "V27", "V28","Amount", "Class"]
CONTINUOUS_COLUMNS = ["Time", "V1", "V2", "V3", "V4", "V5",
        "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13",
        "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21",
        "V22", "V23", "V24", "V25", "V26", "V27", "V28","Amount"]

df_train = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True, engine="python")
df_test = pd.read_csv(test_file, names=COLUMNS, skipinitialspace=True, engine="python")
df_predict = pd.read_csv(predict_file, names=COLUMNS, skipinitialspace=True, engine="python")
df_validation = pd.read_csv(validation_file, names=COLUMNS, skipinitialspace=True, engine="python")

df_train[LABEL_COLUMN] = df_train["Class"].astype(int)
df_test[LABEL_COLUMN] = df_test["Class"].astype(int)
df_predict[LABEL_COLUMN] = df_predict["Class"].astype(int)
df_validation[LABEL_COLUMN] = df_validation["Class"].astype(int)

print(df_train[LABEL_COLUMN])
def input_fn(df):
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  feature_cols = dict(continuous_cols.items())
  label = tf.constant(df[LABEL_COLUMN].values)
  return feature_cols, label

def train_input_fn():
    return input_fn(df_train)
def eval_input_fn():
    return input_fn(df_test)
def predict_input_fn():
    return input_fn(df_predict)
def validation_input_fn():
    return input_fn(df_validation)

m.fit(input_fn=train_input_fn, steps=200)
results = m.evaluate(input_fn=eval_input_fn, steps=1)
for key in sorted(results):
    print("%s: %s" % (key, results[key]))

pred_proba = m.predict_classes(input_fn=lambda: input_fn(df_predict))

for x in pred_proba:
    print(x)
