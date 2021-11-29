import pandas as pd
import json


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

    return data

data = load_data()

data.to_csv("Kaggle Dataset/US_youtube_trending_data.csv")
