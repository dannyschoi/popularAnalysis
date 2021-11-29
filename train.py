import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.optimizers import RMSprop

import streamlit as st

st.set_page_config(layout="wide")
st.title("Youtube Trending Popularity Analysis")


def split_tags(x):
    txt_list = []
    if len(x) > 1:
        for txt in x:
            txt_list.append(txt)
        return " ".join(txt_list)
    else:
        return ""


def load_data():
    data = pd.read_csv("Kaggle Dataset/US_youtube_trending_data.csv")
    with open("Kaggle Dataset/US_category_id.json", "r") as f:
        category = json.load(f)
    f.close()

    cid_list = {}
    for c in category["items"]:
        cid = int(c["id"])
        snippet = c["snippet"]
        if cid not in cid_list:
            cid_list[cid] = snippet["title"]

    expandedCol = ["trending_date", "title", "channelTitle", "categoryId", "tags", "view_count", 'likes', 'dislikes']

    data = data[expandedCol]
    data["category"] = data.categoryId.apply(lambda x: cid_list[x] if x in cid_list else "Unknown")
    data.drop("categoryId", axis=1, inplace=True)

    data["trending_date"] = pd.to_datetime(data.trending_date)
    data["trending_date"] = data["trending_date"].dt.date
    data = data.rename(columns={"trending_date": "datetime"})
    data["split"] = data.tags.apply(lambda x: x.split("|"))
    data["tags"] = data.split.apply(split_tags)
    data.drop("split", axis=1, inplace=True)

    return data


usData = load_data()

us_view = usData.pivot_table(index="category",
                             columns="datetime",
                             values="view_count",
                             aggfunc="sum",
                             fill_value=0)
us_view = us_view.T

us_like = usData.pivot_table(index="category",
                             columns="datetime",
                             values="likes",
                             aggfunc="sum",
                             fill_value=0)
us_like = us_like.T

us_dislike = usData.pivot_table(index="category",
                                columns="datetime",
                                values="dislikes",
                                aggfunc="sum",
                                fill_value=0)
us_dislike = us_dislike.T

frames = [us_view, us_like, us_dislike]
k = ['views', 'likes', 'dislikes']
categories = sorted(usData.category.unique())
us_model = pd.concat(frames, keys=k, axis=1, join='inner').swaplevel(1, 0, axis=1).sort_index(axis=1)

target_column = 'views'
shift_days = 14
df_target = us_model.swaplevel(1, 0, axis=1).sort_index(axis=1)
df_target = df_target[target_column][categories].shift(-shift_days)

x_data = us_model.values[0:-shift_days]
y_data = df_target.values[:-shift_days]

num_train = int(len(x_data) * 0.9)
num_test = len(x_data) - num_train
x_train, x_test = x_data[0:num_train], x_data[num_train:]
y_train, y_test = y_data[0:num_train], y_data[num_train:]

num_x_signals = x_data.shape[1]
num_y_signals = y_data.shape[1]

x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_test_scaled = x_scaler.transform(x_test)
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

validation_data = (np.expand_dims(x_test_scaled, axis=0),
                   np.expand_dims(y_test_scaled, axis=0))


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train_scaled[idx:idx + sequence_length]

        yield (x_batch, y_batch)

generator = batch_generator(batch_size=75,
                            sequence_length=28)
x_batch, y_batch = next(generator)

model = Sequential()
model.add(GRU(units=256,
              return_sequences=True,
              input_shape=(None, num_x_signals,)))
model.add(Dense(num_y_signals, activation='sigmoid'))

if False:
    from tensorflow.python.keras.initializers import RandomUniform

    # Maybe use lower init-ranges.
    init = RandomUniform(minval=-0.05, maxval=0.05)

    model.add(Dense(num_y_signals,
                    activation='linear',
                    kernel_initializer=init))

optimizer = RMSprop(lr=1e-3)
model.compile(optimizer=optimizer, loss="mse")

model.fit(x=generator,
          epochs=20,
          steps_per_epoch=100,
          validation_data=validation_data)

result = model.evaluate(x=np.expand_dims(x_test_scaled, axis=0),
                        y=np.expand_dims(y_test_scaled, axis=0))

print("loss (test-set):", result)
model.save('model')
