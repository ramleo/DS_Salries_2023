from Library import user_defined

import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)

model = joblib.load('models/pipeline_varyweights_rfc')


@app.route('/')
def home():
    print('Rendering index.html')
    return render_template('index.html')


print('after rendering index.html')


@app.route('/predict', methods=['POST'])
def prediction():
    '''
    For rendering results on HTML GUI
    '''

    datapoint_int = [int(x) for x in request.form.values()]
    _, _, columns = user_defined.required_columns()
    print(columns)
    datapoint = np.array(datapoint_int).reshape(1, 7)
    print(datapoint)
    print(datapoint.shape)
    datapoint = pd.DataFrame(datapoint, columns=columns)

    prediction = model.named_steps.rfc.predict(datapoint)

    inc_grp = {1: 'High_Income', 2: 'Low_Income', 3: 'Medium_Income'}

    for i in inc_grp.keys():
        if prediction == i:
            output = i

    return render_template('result.html', prediction_text=f'Employe belongs to {inc_grp[output]} group')


if __name__ == "__main__":
    app.run(debug=True)
