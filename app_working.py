
import requests
import pandas
import simplejson as json
from flask import Flask,render_template,request,redirect,session
import datetime as dt
import joblib


app = Flask(__name__)


 # ==================================get the API Key=========================================== 
api_key = "AIzaSyCuyNRuhz8ckHwGBHPrTdYY4WyFj3YFfko"
from apiclient.discovery import build
youtube = build('youtube', 'v3', developerKey=api_key)






@app.route('/')    
def main():
  return redirect('/index')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')
    
@app.route('/graph', methods=['POST'])
def graph():   
    
    
    import pandas as pd
    import numpy as np

# =====================================Inputs from webpage========================================
    Channel_ID = request.form['channelid']
    new = request.form['content']
    time_duration = request.form['time_dur']
 
    
    if Channel_ID == 'UCraOIV5tXbWQtq7ORVOG4gg':
        input_new = [new+" "+time_duration]    
        pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(input_new)).todense(), 
                 columns = count_vect.get_feature_names())
        
        views_out = int(1.5*int(regressor1.predict(pred_input)))
        likes_out = int(1.5*int(regressor2.predict(pred_input)))
        dislikes_out = int(1.5*int(regressor3.predict(pred_input)))
        

        return render_template('justcurious.html',tables=[df_estimate.to_html(classes='male'), test_dict.to_html(classes='female'), message.to_html(classes='male')],
                          titles = ['na', 'Estimations', 'Test Cases', 'Message'])
    
    
# =============================================================================
#         return render_template('index - 1.html',tables=[df_estimate.to_html(classes='male'), test_dict.to_html(classes='female'), message.to_html(classes='male')],
#                           titles = ['na', 'Estimations', 'Test Cases', 'Message'])
#     
# =============================================================================
    
    
    
# ===============================Show error for invalid input=====================================    
    if Channel_ID == '' or new == '' or time_duration == '':
        return render_template('success.html')

# ======================================get the channel info======================================  
    def get_channel_videos(channel_id):
        res = youtube.channels().list(id=channel_id, 
                                    part='contentDetails').execute()
        playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        videos  = []
        next_page_token = None
        
        while 1:
            res = youtube.playlistItems().list(playlistId=playlist_id,
                                      part='snippet',
                                      maxResults=50,
                                      pageToken=next_page_token).execute()
            videos +=res['items']
            next_page_token = res.get('nextPageToken')
            
            if next_page_token is None:
                break
                
        return videos
    
# ==============Get some stats for each video (likes, dislikes, views, publish date)=====================

    def  get_videos_stats(video_ids):
        stats = []
        for i in range(0, len(video_ids), 50):
            res = youtube.videos().list(id=','.join(video_ids[i:i+50]),
                                 part='statistics').execute()
            stats +=res['items']
            
        return stats       
    
    
# ==============Calling function to get data for SNL channel using SNL channel ID========================
    
    videos = get_channel_videos(Channel_ID)

    
    video_title = list(map(lambda x:x['snippet']['title'], videos))
    video_id = list(map(lambda x:x['snippet']['resourceId']['videoId'], videos))
    video_desc = list(map(lambda x:x['snippet']['description'].split('#')[0], videos))
    
# ==========Calling function to get stats for videos available on SNL channel ID=========================
    
    stats = get_videos_stats(video_id)
    
# ===============Get title and video ID for all available videos on SNL channel==========================
    
    published_date = list(map(lambda x:x['snippet']['publishedAt'].split('T')[0], videos))
    video_views = list(map(lambda x:x['statistics']['viewCount'], stats))
    video_likes = list(map(lambda x:x['statistics']['likeCount'], stats))
    video_dislikes = list(map(lambda x:x['statistics']['dislikeCount'], stats))
    
# ==============================Crating DataFrame from all data==========================================
    #df.drop(df.index, inplace=True)
    df = pd.DataFrame(list(zip(video_title, video_id, video_desc, published_date, video_views, video_likes, video_dislikes)),
                     columns =['video_title', 'video_id', 'video_desc', 'published_date', 'views',
                               'likes', 'dislikes'])
    
    df['views'] = pd.to_numeric(df['views'])
    df['likes'] = pd.to_numeric(df['likes'])
    df['dislikes'] = pd.to_numeric(df['dislikes'])
    
    df['published_date'] = pd.to_datetime(df['published_date'])
    cur_time = pd.to_datetime('20200229', format='%Y%m%d', errors='ignore')
    df['nb_months'] = ((cur_time - df.published_date )/np.timedelta64(1, 'M'))
    df["nb_months"] = df["nb_months"].astype(int)
    
# ==========================Natural Language Processing/creating bag of words=============================
    
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    bag_of_words = []
    for i in range(len(df.video_title)):
        video_description = re.sub('[^a-zA-Z]', ' ', df.video_title[i])
        video_description = video_description.lower()
        video_description = video_description.split()
        ps = PorterStemmer()
        video_description = [ps.stem(word) for word in video_description if not word in set(stopwords.words('english'))]
        video_description = ' '.join(video_description)
        bag_of_words.append(video_description)
    
    
    
# ==========================Add months to the input and create Bag of Words model=================================
    month_list = list(df.nb_months)
    month_str = [str(item) for item in month_list]
    new_bow = [i + " " + j for i, j in zip(bag_of_words, month_str)] 
    
# ==========================================Onehot coder==========================================================    
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    
    count_vect = CountVectorizer(max_features = 1500)
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(new_bow)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    model_input = pd.DataFrame(X_train_tfidf.todense(), columns = count_vect.get_feature_names())

    
# ====================================Algorithm selection for preditction==================================    
    from sklearn.ensemble import RandomForestRegressor
        
    
# ====================================One hot encoder fot future caption===================================  
    input_new = [new+" "+time_duration]    
    pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(input_new)).todense(), 
                 columns = count_vect.get_feature_names())
    
    y1 = df['views'].values
    regressor1 = RandomForestRegressor(n_estimators = 600, random_state = 0)
    regressor1.fit(model_input, y1)
    views_out = int(1.5*int(regressor1.predict(pred_input)))
    
    y2 = df['likes'].values
    regressor2 = RandomForestRegressor(n_estimators = 600, random_state = 0)
    regressor2.fit(model_input, y2)
    likes_out = int(1.5*int(regressor2.predict(pred_input)))
    
    
    y3 = df['dislikes'].values
    regressor3 = RandomForestRegressor(n_estimators = 600, random_state = 0)
    regressor3.fit(model_input, y3)
    dislikes_out = int(1.5*int(regressor3.predict(pred_input)))
    
    
    
# =================================================Save the model================================================    
   
    
    filename = 'finalized_model.sav'
    joblib.dump(model, filename)

# ====================================Create dataframe to be shown as table in graph()==================================   
    df_estimate = pd.DataFrame({'New_Title': [new],
           'Estimated_Views': [views_out], 'Estimated_Likes': [likes_out], 'Estimated_Dislikes': [dislikes_out]})

# ====================================test case for first 10================================================
    test_new = []
    test_title = []
    test_views = []
    test_likes = []
    test_dislikes = []
    test_dict = {}
    
    for i in range(0,10):
        test_new.append([df['video_title'][i+106]+" "+str(df['nb_months'][i+106])])   
        pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(test_new[i])).todense(), 
                     columns = count_vect.get_feature_names())
        views_out = int(int(regressor1.predict(pred_input)))
        likes_out = int(int(regressor2.predict(pred_input)))
        dislikes_out = int(int(regressor3.predict(pred_input)))
        
        test_title.append(df['video_title'][i+106])
        test_views.append(str(int(abs(100*(df['views'][i+106]-views_out)/df['views'][i+106])))+' %')
        test_likes.append(str(int(abs(100*(df['likes'][i+106]-likes_out)/df['likes'][i+106])))+' %')
        test_dislikes.append(str(int(abs(100*(df['dislikes'][i+106]-dislikes_out)/df['dislikes'][i+106])))+' %') 
        
# =============================================================================
#         
#     for i in range(5,10):
#         test_new.append([df['video_title'][i+127]+" "+str(df['nb_months'][i+127])])   
#         pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(test_new[i])).todense(), 
#                      columns = count_vect.get_feature_names())
#         views_out = int(int(regressor1.predict(pred_input)))
#         likes_out = int(int(regressor2.predict(pred_input)))
#         dislikes_out = int(int(regressor3.predict(pred_input)))
#          
#  
#         test_title.append(df['video_title'][i+127])
#         test_views.append(str(int(abs(100*(df['views'][i+127]-views_out)/df['views'][i+127])))+' %')
#         test_likes.append(str(int(abs(100*(df['likes'][i+127]-likes_out)/df['likes'][i+127])))+' %')
#         test_dislikes.append(str(int(abs(100*(df['dislikes'][i+127]-dislikes_out)/df['dislikes'][i+127])))+' %') 
#         
# =============================================================================
        
    
# =============================================================================
#     for i in range(0,5):
#         test_new.append([df['video_title'][i+601]+" "+str(df['nb_months'][i+601])])   
#         pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(test_new[i])).todense(), 
#                      columns = count_vect.get_feature_names())
#         views_out = int(int(regressor1.predict(pred_input)))
#         likes_out = int(int(regressor2.predict(pred_input)))
#         dislikes_out = int(int(regressor3.predict(pred_input)))
#         
# 
#         test_title.append(df['video_title'][i+601])
#         test_views.append(str(int(abs(100*(df['views'][i+601]-views_out)/df['views'][i+601])))+' %')
#         test_likes.append(str(int(abs(100*(df['likes'][i+601]-likes_out)/df['likes'][i+601])))+' %')
#         test_dislikes.append(str(int(abs(100*(df['dislikes'][i+601]-dislikes_out)/df['dislikes'][i+601])))+' %') 
#         
#     for i in range(5,10):
#         test_new.append([df['video_title'][i+127]+" "+str(df['nb_months'][i+127])])   
#         pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(test_new[i])).todense(), 
#                      columns = count_vect.get_feature_names())
#         views_out = int(int(regressor1.predict(pred_input)))
#         likes_out = int(int(regressor2.predict(pred_input)))
#         dislikes_out = int(int(regressor3.predict(pred_input)))
#         
# 
#         test_title.append(df['video_title'][i+127])
#         test_views.append(str(int(abs(100*(df['views'][i+127]-views_out)/df['views'][i+127])))+' %')
#         test_likes.append(str(int(abs(100*(df['likes'][i+127]-likes_out)/df['likes'][i+127])))+' %')
#         test_dislikes.append(str(int(abs(100*(df['dislikes'][i+127]-dislikes_out)/df['dislikes'][i+127])))+' %') 
# 
# =============================================================================

        

 
# ====================================Create dataframe to be shown as table in graph()==================================      

    
    test_dict = pd.DataFrame({'Prediction Error in %': test_title,
           'Views': test_views, 'Likes': test_likes, 'Dislikes': test_dislikes})

    message = pd.DataFrame({'Messege for you': ['Dude you can do better than this! Under construction...']})
    
# ==========================================Return dataframes in table format in the webpage===========================
    
    return render_template('index - 1.html',tables=[df_estimate.to_html(classes='male'), test_dict.to_html(classes='female'), message.to_html(classes='male')],
    titles = ['na', 'Estimations', 'Test Cases', 'Message'])
    
  

if __name__ == '__main__':
    app.run(port=33507)
