import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import numpy as np
from tensorflow import keras
import wordcloud as wordcloud
import datetime as dt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

st.title("Youtube Trending Popularity Analysis")
st.markdown("""
This app provides analytical visualizations of Youtube's Trending Videos.
\nThe left sidebar can be changed in order to change the range of data.

""")
st.sidebar.header('Data Filters')
today = dt.date(2021, 11, 17)
past = today - dt.timedelta(days=7)
start_date = st.sidebar.date_input('Start date', past, min_value=dt.date(2020, 8, 12), max_value=dt.date(2021, 11, 17))
end_date = st.sidebar.date_input('End date', today, min_value=dt.date(2020, 8, 12), max_value=dt.date(2021, 11, 17))
if start_date <= end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')


def split_tags(x):
    txt_list = []
    if len(x) > 1:
        for txt in x:
            txt_list.append(txt)
        return " ".join(txt_list)
    else:
        return ""


@st.cache
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

df_selected = usData
# Filtering data
mask = (df_selected["datetime"] >= start_date) & (df_selected["datetime"] <= end_date)

# Category Filter
uniqueCate = sorted(df_selected.category.unique())
selectCate = st.sidebar.multiselect('Video Category', uniqueCate, uniqueCate)
cloudCate = st.sidebar.selectbox('WordCloud Category', selectCate)
df_selected = usData[(usData.category.isin(selectCate))]
df_selected = df_selected.loc[mask]

st.header('')
with st.expander(
        'Data Dimension: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.'):
    st.dataframe(df_selected)


def viewsDates(df):
    x = df.groupby("datetime").mean(numeric_only=True).loc[:, ["view_count"]]
    return x


with st.expander("Total view count over time"):
    st.line_chart(viewsDates(df_selected))


def viewsByCate(df):
    category = df.category.unique()
    if category.size < 13:
        fig, axes = plt.subplots(4, 3, figsize=(12, 12))
    else:
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    ax = axes.ravel()
    dateFormat = mdates.DateFormatter('%y/%m/%d')

    for i, cate in enumerate(category):
        x = df[df.category == cate].groupby("datetime").mean(numeric_only=True).loc[:, ["view_count"]]
        x.plot(ax=ax[i])
        ax[i].set_title(cate)
        plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right')
        ax[i].xaxis.set_major_formatter(dateFormat)

    plt.tight_layout()
    return fig


def categoryCloud(df):
    tags = df['tags']
    wc = wordcloud.WordCloud(width=1200, height=800,
                             collocations=False, background_color="white",
                             colormap="tab20b").generate(" ".join(tags))
    fig = plt.figure(figsize=(15, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")

    return fig


with st.expander('View count over time by Category'):
    st.pyplot(viewsByCate(df_selected))

with st.expander('WordCloud of Selected Category'):
    df_cloud = df_selected[df_selected.category.isin([cloudCate, ''])]
    st.pyplot(categoryCloud(df_cloud))

saved_model = keras.models.load_model("model")

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

future_x = us_model.values[len(us_model) - 14:]
x_future_scaled = x_scaler.fit_transform(future_x)


def plot_comparison(start_idx, length=100, train=True):
    if train:
        x = x_train_scaled
        y_true = y_train
        start = dt.date(2020, 8, 12)
    else:
        x = x_test_scaled
        y_true = y_test
        start = dt.date(2020, 11, 17) - dt.timedelta(days=44)

    end_idx = start_idx + length

    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    x = np.expand_dims(x, axis=0)

    y_pred = saved_model.predict(x)

    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    fig, axes = plt.subplots(15, 1, figsize=(15, 60))
    ax = axes.ravel()

    start = start + dt.timedelta(days=start_idx)
    dateRange = pd.date_range(start, periods=length)

    for signal in range(len(categories)):
        signal_pred = y_pred_rescaled[:, signal]

        signal_true = y_true[:, signal]

        ax[signal].plot(dateRange, signal_true, label='true')
        ax[signal].plot(dateRange, signal_pred, label='Predictive')

        ax[signal].set_ylabel(categories[signal])
        ax[signal].legend()

    return fig


with st.expander('View count Predictive Modelling with train data'):
    st.pyplot(plot_comparison(100, length=100, train=True))

with st.expander('View count Predictive Modelling with test data'):
    st.pyplot(plot_comparison(0, length=40, train=False))


def future_plot(x_future_scaled):
    x_future_scaled = np.expand_dims(x_future_scaled, axis=0)
    y_future = saved_model.predict(x_future_scaled)
    start = dt.date(2020, 11, 17)
    dateRange = pd.date_range(start, periods=14)
    y_future_rescaled = y_scaler.inverse_transform(y_future[0])
    fig, axes = plt.subplots(15, 1, figsize=(15, 60))
    ax = axes.ravel()
    for signal in range(len(categories)):
        signal_f = y_future_rescaled[:, signal]
        ax[signal].plot(dateRange, signal_f, label='Future')

        ax[signal].set_ylabel(categories[signal])

    return fig


with st.expander('View count Predictive Modelling- Future 2 weeks'):
    st.pyplot(future_plot(x_future_scaled))
