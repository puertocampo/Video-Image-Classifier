# -*- coding=utf-8 -*-

from flask import Flask, render_template, request, redirect, url_for, g
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import json
import base64
from PIL import Image
from io import BytesIO
import sqlite3
import numpy as np
from datetime import datetime as dt
from pytz import timezone
import copy

app = Flask(__name__)

#result.htmlに渡す画像データ、history.htmlに渡す履歴データをグローバル変数に格納
current_image_classify_result = ''
current_image_jpeg_base64 = ''
image_classify_result_list = []
image_jpeg_base64_list = []
timestamp_list = []

def list_FIFO(array,append_element):#要するにキューの実装。array.pop(0)を使ってたらなぜかメモリ不足になったので・・・。
    if len(array)>=5:
        temp_array = copy.deepcopy(array[-4:])
    else:
        temp_array = copy.deepcopy(array)
    del array #メモリを解放（←できてんのか？）
    temp_array.append(append_element) #キューに追加
    return temp_array

@app.route('/')
def index():
    return render_template('index.html')

#/image_classifyでは"画像受け取る→グローバル変数を更新→変換してVGGに投げる→ニューラルネット→結果を円グラフ用に整形してグローバル変数を更新"の流れ
@app.route('/image_classify', methods=['POST'])
def image_classify():
    if request.method == 'POST':
        enc_data  = request.form['img'] # enc_dataはu"data:image/png;base64,kpvdfkgspkhgsKOJ..."の形(base64)
        global current_image_jpeg_base64
        current_image_jpeg_base64 = enc_data
        global image_jpeg_base64_list
        image_jpeg_base64_list = list_FIFO(image_jpeg_base64_list, current_image_jpeg_base64)
        global timestamp_list
        timestamp_list = list_FIFO(timestamp_list,dt.now(timezone('Asia/Tokyo')).strftime("%Y/%m/%d %H:%M:%S"))
        dec_data = base64.b64decode( enc_data.split(',')[1] ) # "data:image/png;base64"以下を取り出す

        #==以下kerasによるimage classification==
        img = Image.open(BytesIO(dec_data))
        img_resized = img.resize((224, 224)).convert('RGB')#RGBA画像をRGB画像に変換、画素数も統一
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        model = VGG16(weights='imagenet')
        preds = model.predict(preprocess_input(x))
        results = decode_predictions(preds, top=5)[0]
        #==以上image classification==

        #以下円グラフ用のjsonデータ整形
        colors_list = ["#9acce3","#70b062","#dbdf19","#a979ad","#cd5638"]
        highlight_list = ["#aadbf2","#7fc170","#ecef23","#bb8ebf","#e2694a"]
        label_Prob_Json = []
        others_value = 100
        for i,result in enumerate(results):
            mini_dict = {'value':"{0:0.2f}".format(round(result[2]*100,2)),'color':colors_list[i],'highlight':highlight_list[i],'label':result[1].upper().replace('_',' ')}
            label_Prob_Json.append(mini_dict)
            others_value-=result[2]*100
        label_Prob_Json.append({'value':"{0:0.2f}".format(round(others_value,2)),'color':'#000','highlight':'#000','label':'OTHERS'})
        global current_image_classify_result
        current_image_classify_result = label_Prob_Json
        global image_classify_result_list
        image_classify_result_list = list_FIFO(image_classify_result_list, current_image_classify_result)
        return render_template('index.html') #返り値は必ず必要、なかったら怒られる ここでrender_template('result.html')したいけど

@app.route('/result')
def result():
    return render_template('result.html',label_Prob_Dict=current_image_classify_result,image_raw_data=current_image_jpeg_base64)

@app.route('/history')
def history():
    history_image_result_json = {'result':image_classify_result_list, 'data':image_jpeg_base64_list, 'date':timestamp_list}
    return render_template('history.html',history_image_result_json=history_image_result_json)

@app.errorhandler(404)
def page_not_found(e):
    return 'Sorry, Nothing at this URL.', 404

@app.errorhandler(500)
def application_error(e):
    return 'Sorry, unexpected error: {}'.format(e), 500

if __name__=='__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
