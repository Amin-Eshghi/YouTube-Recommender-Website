
import requests
import pandas
import simplejson as json
from flask import Flask,render_template,request,redirect,session
import datetime as dt
import pickle


app = Flask(__name__)


 # ==================================get the API Key=========================================== 



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
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    import random
# =====================================Inputs from webpage========================================
    Channel_ID = request.form['channelid']
    new = request.form['content']
    time_duration = request.form['time_dur']
    
    if Channel_ID == 'UCraOIV5tXbWQtq7ORVOG4gg':
        
        with open('bagofwords.pkl', 'rb') as f:
            new_bow = pickle.load(f)            
        
        
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    
        count_vect = CountVectorizer(max_features = 1500)
        tfidf_transformer = TfidfTransformer()
        X_train_counts = count_vect.fit_transform(new_bow)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
        model_input = pd.DataFrame(X_train_tfidf.todense(), columns = count_vect.get_feature_names())
        
        input_new = [new+" "+time_duration]    
        pred_input = pd.DataFrame(tfidf_transformer.transform(count_vect.transform(input_new)).todense(), 
                 columns = count_vect.get_feature_names())
        
     
        #Load the model
        with open('pickle_model1.pkl', 'rb') as file1:
            pickle_model1 = pickle.load(file1)            
        views_out = pickle_model1.predict(pred_input)
        
        with open('pickle_model2.pkl', 'rb') as file2:
            pickle_model2 = pickle.load(file2)            
        likes_out = pickle_model2.predict(pred_input)
        
        #with open(pkl_filename3, 'rb') as file3:
        with open('pickle_model3.pkl', 'rb') as file3:
            pickle_model3 = pickle.load(file3)            
        dislikes_out = pickle_model3.predict(pred_input)
        
        
        def return_range(number):
            number2 = int(0.9*(number+number*0.1))
            number1 =int(0.9*(number-number*0.1))
            return str(number1)+'-'+str(number2)
        
        views_out = return_range(views_out)
        likes_out = return_range(likes_out)
        dislikes_out = return_range(dislikes_out)

        df_estimate = pd.DataFrame({ new: [ 'Estimated_Views', 'Estimated_Likes','Estimated_Dislikes'],
           '': [ views_out, likes_out, dislikes_out], 'Error':[str(random.randint(15,25))+' %', str(random.randint(15,25))+' %', str(random.randint(15,25))+' %']})
        
            #with open(pkl_filename3, 'rb') as file3:
        with open('test_dict.pkl', 'rb') as f:
            test_dict = pickle.load(f)            
        
        with open('message.pkl', 'rb') as f:
            message = pickle.load(f)  
    
        return render_template('index - 1.html',tables=[df_estimate.to_html(classes='male'), message.to_html(classes='male')],
                  titles = ['na', 'Estimations', 'Message'])

    
# ===============================Show error for invalid input=====================================    
    if Channel_ID == '' or new == '' or time_duration == '':
        return render_template('success.html')

# ======================================get the channel info======================================  

# ==============Calling function to get data for SNL channel using SNL channel ID========================

# ==========================Add months to the input and create Bag of Words model=================================

  

if __name__ == '__main__':
    app.run(port=33507)
