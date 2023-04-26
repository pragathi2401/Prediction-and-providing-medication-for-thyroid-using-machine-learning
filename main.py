from os import O_TRUNC
from flask import Flask,render_template,request,make_response, redirect, url_for,abort
import requests
import pickle
import numpy as np
from markupsafe import escape

app = Flask(__name__)

with open("thyroid_detection_gbc_model.pkl","rb") as model_file:
    model=pickle.load(model_file)

@app.route('/')
def index():
   return render_template('home.html')

@app.route("/moreinfo", methods = ["GET", "POST"])
def moreinfo():
    return render_template('moreinfo.html')

@app.route("/predict", methods = ["GET", "POST"])
def predict():
    return render_template('predict.html')

@app.route("/predictresult", methods = ["GET", "POST"])
def predictresult():
    if request.method == "POST":
        age = int(request.form['age'])  
        sex= request.form.get('sex')
        on_thyroxine= request.form.get('on_thyroxine')
        on_antithyroid_meds=request.form.get('on_antithyroid_meds')
        pregnant=request.form.get('pregnant')
        thyroid_surgery = request.form.get('thyroid_surgery')
        tumor = request.form.get('tumor')
        TSH = float(request.form['TSH'])
        T3 = float(request.form['T3'])
        TT4 = float(request.form['TT4'])
        T4U = float(request.form['T4U'])
        FTI = float(request.form['FTI'])


        
        if sex=="Male":
            sex=1.0
        else:
            sex=0.0
        
        if on_thyroxine=="True":
            on_thyroxine=1
        else:
            on_thyroxine=0
        if on_antithyroid_meds=="True":
            on_antithyroid_meds=1
        else:
            on_antithyroid_meds=0
        if pregnant=="True":
            pregnant=1
        else:
            pregnant=0
      
        if thyroid_surgery=="True":
            thyroid_surgery=1
        else:
            thyroid_surgery=0
        
       
        if tumor=="True":
            tumor=1
        else:
            tumor=0

        # if sex==1 and pregnant==1:
        #     return
        # else:
        arr=np.array([[age,sex,on_thyroxine,on_antithyroid_meds,pregnant,thyroid_surgery,tumor
                       ,TSH,T3,TT4,T4U,FTI]])
        pred=model.predict(arr)

        
        if pred==0:
            res_val="""
           <div class="box"> <h1><span style="color:green;">Congratulations ! You DON'T have Thyroid Disease.</span></h1></div>
				
            """
        elif pred==1:
            res_val="""
             <div class="box"><h2><span style="color:red;">OOPs ! You have Hyperthyroid Disease.</span></h2><br>
				<p><b>Recommendations</b> <span><br>
	    1. Attend regular follow-up appointments with a healthcare provider. Regular monitoring of thyroid function can help ensure that
	    hormone levels are within a healthy range.<br>
            2. Follow a healthy and balanced diet. A well-balanced diet can help support thyroid function and overall health.
            It's important to eat a variety of nutrient-rich foods, including whole grains, fruits, vegetables, lean proteins, and healthy fats.<br>
            3. Avoid iodine-rich foods. Iodine is necessary for the production of thyroid hormone, but consuming too much iodine can worsen hyperthyroidism.
            Foods high in iodine include seaweed, seafood, and iodized salt. It's important to speak with a healthcare provider or registered dietitian
            to determine the appropriate amount of iodine for your specific case.<br>
            4. Manage stress levels. Stress can worsen hyperthyroidism symptoms, so engaging in stress-reducing activities such as meditation,
            yoga, or deep breathing exercises may be helpful.<br>
            5. Consider medication or other treatments. Hyperthyroidism can be treated with medications that reduce the production of thyroid hormone or
             by destroying or removing the thyroid gland. It's important to discuss treatment options with a healthcare provider.<br>
            6. Be aware of potential complications. Hyperthyroidism can lead to complications such as osteoporosis, heart problems,
            and eye problems. Regular monitoring and treatment can help prevent or manage these complications.


</span></p></div>
    """
        elif pred==2:
            res_val="""
        <div class="box"><h2><span style="color:red;">OOPs! You have Primary Hypothyroid Disease.</span></h2><br>
        <p><b>Recommendations</b><span><br>1. Take thyroid hormone replacement medication as prescribed by a healthcare provider.
        The medication replaces the thyroid hormone that the body is not producing enough of and helps to regulate the body's metabolism.<br>
        2. Follow a healthy and balanced diet. A well-balanced diet can help support thyroid function and overall health.
          It's important to eat a variety of nutrient-rich foods, including whole grains, fruits, vegetables, lean proteins, and healthy fats.<br>
        3. Exercise regularly. Regular exercise can help to boost energy levels and support overall health. It's important to check with a healthcare provider before starting an exercise program, especially if you have any other medical conditions.<br>
        4. Manage stress levels. Chronic stress can affect the body's hormonal balance, including the production of thyroid hormone. Engaging in stress-reducing activities such as meditation, yoga, or deep breathing exercises may be helpful.<br>
        5.Attend regular follow-up appointments with a healthcare provider. Thyroid function should be regularly monitored to ensure that hormone levels are within a healthy range.<br>
        6. As well as you can take vitamin B supplements and probiotics.<br>
        </span></p></div>
    """
        
        elif pred==3:
            res_val="""
        <div class="box"><h2><span style="color:red;">OOPs! You have Compensated Hypothyroid Disease.</span></h2><br>
        <p><b>Recommendations</b><span><br>
        1. Attend regular follow-up appointments with a healthcare provider. Even though the thyroid hormone levels may be within the normal range,
        it's important to regularly monitor thyroid function to ensure that the levels remain stable.<br>
        2. Follow a healthy and balanced diet. A well-balanced diet can help support thyroid function and overall health. It's important to eat a variety of nutrient-rich foods, including whole grains, fruits, vegetables, lean proteins, and healthy fats.<br>
        3. Exercise regularly. Regular exercise can help to boost energy levels and support overall health. It's important to check with a healthcare provider before starting an exercise program, especially if you have any other medical conditions.<br>
        4. Manage stress levels. Chronic stress can affect the body's hormonal balance, including the production of thyroid hormone. Engaging in stress-reducing activities such as meditation, yoga, or deep breathing exercises may be helpful.<br>
        5. Monitor for symptoms of hypothyroidism. Even though the thyroid hormone levels may be within the normal range, some individuals with compensated hypothyroidism may still experience symptoms such as fatigue, weight gain, and depression.
           If symptoms persist, it's important to discuss them with a healthcare provider.</span></p></div>
    """

        elif pred==4:
            res_val="""
        <div class="box"><h2><span style="color:red;">OOPs! You have Secondary Hypothyroid Disease.</span></h2><br>
        <p><b>Recommendations</b><span><br>
        1. Take thyroid hormone replacement medication as prescribed by a healthcare provider. The medication replaces the thyroid hormone that the body is not
        producing enough of and helps to regulate the body's metabolism.<br>
        2. Attend regular follow-up appointments with a healthcare provider. Thyroid function should be regularly monitored to ensure that hormone levels are
        within a healthy range.Follow a healthy and balanced diet. A well-balanced diet can help support thyroid function and overall health.
        It's important to eat a variety of nutrient-rich foods, including whole grains, fruits, vegetables, lean proteins, and healthy fats.<br>
        3. Exercise regularly. Regular exercise can help to boost energy levels and support overall health. It's important to check with a healthcare
        provider before starting an exercise program, especially if you have any other medical conditions.<br>
       4. Manage stress levels. Chronic stress can affect the body's hormonal balance, including the production of thyroid hormone.
       Engaging in stress-reducing activities such as meditation, yoga, or deep breathing exercises may be helpful.<br>
       5. Identify and treat any underlying conditions. Secondary hypothyroidism may be caused by an underlying condition such as a pituitary or hypothalamus disorder.
       It's important to identify and treat any underlying conditions to help improve thyroid function.
</span></p></div>
    """
        return render_template('predictresult.html',output=res_val)


    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=False)
