#libraries including

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec


#reading data from datasets
events = input('train_events file path: ')
train_events = pd.read_csv(events)
video = input('video_info file path: ')
video_info = pd.read_csv(video)

print('Starting processing data...')

merged_data = pd.merge(train_events, video_info, on='rutube_video_id', how='left')

#Databases merging
aggregated_data = merged_data.groupby('viewer_uid').agg({
    'event_timestamp' : lambda x : list(x),
    'region' : 'first',
    'ua_device_type' : lambda x : list(x),
    'ua_client_type' : lambda x : list(x),
    'ua_os' : lambda x : list(x),
    'ua_client_name' : lambda x : list(x),
    'total_watchtime' : lambda x : list(x),
    'rutube_video_id' : lambda x : list(x),
    'title' : lambda x : list(x),
    'category' : lambda x : list(x),
    'author_id' : lambda x : list(x)
}).reset_index()


#Dropping useless columns
aggregated_data = aggregated_data.drop(columns=['event_timestamp'])


#Coding columns
label_encoder_region = LabelEncoder()
aggregated_data['region'] = label_encoder_region.fit_transform(aggregated_data['region'])
aggregated_data.head()
aggregated_data['ua_device_type'] = aggregated_data['ua_device_type'].apply(lambda x: x if isinstance(x, list) else eval(x))
aggregated_data['ua_client_type'] = aggregated_data['ua_client_type'].apply(lambda x: x if isinstance(x, list) else eval(x))
aggregated_data['ua_os'] = aggregated_data['ua_os'].apply(lambda x: x if isinstance(x, list) else eval(x))
aggregated_data['ua_client_name'] = aggregated_data['ua_client_name'].apply(lambda x: x if isinstance(x, list) else eval(x))
# Create dictionaries for replacement
device_type_dict = {value: idx for idx, value in enumerate(set([item for sublist in aggregated_data['ua_device_type'] for item in sublist]))}
client_type_dict = {value: idx for idx, value in enumerate(set([item for sublist in aggregated_data['ua_client_type'] for item in sublist]))}
os_dict = {value: idx for idx, value in enumerate(set([item for sublist in aggregated_data['ua_os'] for item in sublist]))}
client_name_dict = {value: idx for idx, value in enumerate(set([item for sublist in aggregated_data['ua_client_name'] for item in sublist]))}
# Function to replace strings with numerical values
def replace_with_dict(lst, replace_dict):
    return [replace_dict[item] for item in lst]
# Replace strings with numerical values
aggregated_data['ua_device_type'] = aggregated_data['ua_device_type'].apply(lambda x: replace_with_dict(x, device_type_dict))
aggregated_data['ua_client_type'] = aggregated_data['ua_client_type'].apply(lambda x: replace_with_dict(x, client_type_dict))
aggregated_data['ua_os'] = aggregated_data['ua_os'].apply(lambda x: replace_with_dict(x, os_dict))
aggregated_data['ua_client_name'] = aggregated_data['ua_client_name'].apply(lambda x: replace_with_dict(x, client_name_dict))
aggregated_data = aggregated_data.drop(columns=['rutube_video_id', 'author_id'])



#Word2Vec model
titles = aggregated_data['title'].apply(lambda x: ' '.join(x)).tolist()
titles_tokenized = [title.split() for title in titles]
model = Word2Vec(sentences=titles_tokenized, vector_size=100, window=5, min_count=1, workers=4)
def get_title_vector(title, model):
    words = title.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return [0] * model.vector_size
aggregated_data['title_vector'] = aggregated_data['title'].apply(lambda x: get_title_vector(' '.join(x), model))
aggregated_data = aggregated_data.drop(columns=['title'])

categories = aggregated_data['category'].apply(lambda x: ' '.join(x)).tolist()
categories_tokenized = [category.split() for category in categories]
category_model = Word2Vec(sentences=categories_tokenized, vector_size=100, window=5, min_count=1, workers=4)
def get_category_vector(category, model):
    words = category.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return [0] * model.vector_size
aggregated_data['category_vector'] = aggregated_data['category'].apply(lambda x: get_category_vector(' '.join(x), category_model))
aggregated_data = aggregated_data.drop(columns=['category'])

output = aggregated_data['viewer_uid']

aggregated_data.to_csv('temporary_data.csv', index=False)

print('Data processing is finished. Preparing data to input to the models...')

#Continuing of data transformation
import numpy as np
file_path = 'temporary_data.csv'
data = pd.read_csv(file_path)
data['title_vector'] = data['title_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
max_length = max(data['title_vector'].apply(len))
data['title_vector'] = data['title_vector'].apply(lambda x: np.pad(x, (0, max_length - len(x)), 'constant'))
data['category_vector'] = data['category_vector'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

#Checking the length of all vectors and bringing them to the same length if necessary
max_length = max(data['category_vector'].apply(len))
data['category_vector'] = data['category_vector'].apply(lambda x: np.pad(x, (0, max_length - len(x)), 'constant'))
data = data.drop(columns=['viewer_uid'])

#Converting DataFrame to numpy array
from collections import Counter
input = []
for i in range(len(data['region'])):
    temp = []
    temp.append(int(data['region'][i]))
    copy = data['ua_device_type'][i].replace('[', '').replace(']', '').split(',')
    for i in range(len(copy)):
        copy[i] = int(copy[i])
    counter_ua_device_type = Counter(copy)
    temp.append(int(counter_ua_device_type.most_common(1)[0][0]))
    copy = data['ua_client_type'][i].replace('[', '').replace(']', '').split(',')
    for i in range(len(copy)):
        copy[i] = int(copy[i])
    counter_ua_client_type = Counter(copy)
    temp.append(int(counter_ua_client_type.most_common(1)[0][0]))
    copy = data['ua_os'][i].replace('[', '').replace(']', '').split(',')
    for i in range(len(copy)):
        copy[i] = int(copy[i])
    counter_ua_os = Counter(copy)
    temp.append(int(counter_ua_os.most_common(1)[0][0]))
    su = 0
    for j in data['total_watchtime'][i].replace('[', ' ').replace(']', ' ').split(','):
        su += int(j)
    temp.append(int(su))
    for j in range(len(data['title_vector'][i])):
        temp.append(float(data['title_vector'][i][j]))
    for k in range(len(data['category_vector'][i])):
        temp.append(float(data['category_vector'][i][k]))
    np_temp = np.array(temp)
    input.append(np_temp)
np_input = np.array(input)

print('Data is ready. Starting prediction...')


#loading the models
import joblib

knn_sex_f = joblib.load('knn_sex_model.pkl')
knn_age_f = joblib.load('knn_age_model.pkl')

#predicting the results
pred_age = knn_age_f.predict(np_input)
pred_sex = knn_sex_f.predict(np_input)

print('Prediction is finished. Saving data to CSV...')

#saving data to CSV
predictions = pd.DataFrame({
    'viewer_uid': output,
    'pred_sex': pred_sex,
    'pred_age': pred_age
})
predictions.to_csv('predictions.csv', index=False)
print('Data is saved to predictions.csv')
