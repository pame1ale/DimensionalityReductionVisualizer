from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)

#Reading data
#data_df = pd.read_csv("static/data/churn_data.csv")
churn_df = pd.read_csv("static/data/wine.csv")
commu = pd.read_csv("static/data/communities.csv")
air = pd.read_csv("static/data/airqualityuci.csv")

#churn_df = data_df[(data_df['Churn']=="Yes").notnull()]

@app.route('/')
def index():
   return render_template('index.html')

def calculate_percentage(val, total):
   """Calculates the percentage of a value over a total"""
   np.seterr(invalid='ignore')
   percent = np.round((np.divide(val, total) * 100), 2)
   return percent

def data_creation(data, percent, class_labels, group=None):
   for index, item in enumerate(percent):
       data_instance = {}
       data_instance['category'] = class_labels[index]
       data_instance['value'] = item
       data_instance['group'] = group
       data.append(data_instance)

@app.route('/get_piechart_data')
def get_piechart_data():
   contract_labels = ['1', '2', '3']
   _ = churn_df.groupby('class').size().values
   class_percent = calculate_percentage(_, np.sum(_)) #Getting the value counts and total

   piechart_data= []
   data_creation(piechart_data, class_percent, contract_labels)
   return jsonify(piechart_data)

@app.route('/get_barchart_data')
def get_barchart_data():
   tenure_labels = ['10-10.5', '11-11.5', '12-12.5', '13-13.5', '14-14.5', '15-15.5', '16-16.5', '17-17.5']
   churn_df['alcohol_group'] = pd.cut(churn_df.alcohol, range(0, 81, 10), labels=tenure_labels)
   select_df = churn_df[['alcohol_group','class']]
   contract_month = select_df[select_df['class']=='1']
   contract_one = select_df[select_df['class']=='2']
   contract_two =  select_df[select_df['class']=='3']
   _ = contract_month.groupby('alcohol_group').size().values
   mon_percent = calculate_percentage(_, np.sum(_))
   _ = contract_one.groupby('alcohol_group').size().values
   one_percent = calculate_percentage(_, np.sum(_))
   _ = contract_two.groupby('alcohol_group').size().values
   two_percent = calculate_percentage(_, np.sum(_))
   _ = select_df.groupby('alcohol_group').size().values
   all_percent = calculate_percentage(_, np.sum(_))

   barchart_data = []
   data_creation(barchart_data,all_percent, tenure_labels, "All")
   data_creation(barchart_data,mon_percent, tenure_labels, "1")
   data_creation(barchart_data,one_percent, tenure_labels, "2")
   data_creation(barchart_data,two_percent, tenure_labels, "3")
   return jsonify(barchart_data)




if __name__ == '__main__':
   app.run(debug=True)