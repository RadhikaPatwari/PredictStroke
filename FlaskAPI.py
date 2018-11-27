from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.externals import joblib
import os, io, csv
import pickle
from bokeh.plotting import figure, output_file, show
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource

app = Flask(__name__, template_folder='templates')


@app.route('/predict')
def callapi():
    # test_json=request.get_json()
    # test = pd.read_json(test_json, orient='records')

    return render_template('hello.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        f = request.files['file']
        if not f:
            return "No file"

        file_data = f.read()

        testdata = io.StringIO(unicode(file_data))

        df = pd.read_csv(testdata, sep=",")

        df1 = df[:]

        feature_cols = ['age', 'hypertension', 'heart_disease', 'Residence_type', 'avg_glucose_level', 'bmi',
                        'smoking_status']

        df = df.replace({'gender': {'Female': 0, 'Male': 1}})
        df = df.replace({'Residence_type': {'Rural': 0, 'Urban': 1}})
        df = df.replace({'smoking_status': {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2}})

        print(df)

        loadedModel = pickle.load(open('./stroke.pkl', 'rb'))
        result_pred = loadedModel.predict(df[feature_cols])

        # prediction_series=list(pd.Series(result_pred))
        # failureColumn=df['failure']
        # final_predictions=pd.DataFrame(list(zip(failureColumn, prediction_series)))
        # responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        # responses.status_code = 200

        df1['predicted_score'] = result_pred

        # output_file("bars.html")
        stroke_values = df['stroke'].unique()
        strokevalues = []
        stroke_counts = []
        for e in stroke_values:
            strokevalues.append(str(e))
            count = df[df['stroke'] == e]['stroke'].count()
            stroke_counts.append(count)

        p = figure(x_range=strokevalues, plot_height=250, title="Stroke Counts",
                   toolbar_location=None, tools="")

        p.vbar(x=strokevalues, top=stroke_counts, width=0.9)

        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        # show(p)
        script1, div1 = components(p)

        predicted_stroke_values = df1['predicted_score'].unique()
        predicted_strokevalues = []
        predicted_stroke_counts = []
        for e in predicted_stroke_values:
            predicted_strokevalues.append(str(e))
            predicted_count = df1[df1['predicted_score'] == e]['predicted_score'].count()
            predicted_stroke_counts.append(predicted_count)

        q = figure(x_range=predicted_strokevalues, plot_height=250, title="Predicted Stroke Counts",
                   toolbar_location=None, tools="")

        q.vbar(x=predicted_strokevalues, top=predicted_stroke_counts, width=0.9)

        q.xgrid.grid_line_color = None
        q.y_range.start = 0

        script2, div2 = components(q)

        return render_template('result.html', result=df1.to_html(), the_div1=div1, the_script1=script1, the_div2=div2,
                               the_script2=script2)


if __name__ == '__main__':
    app.run(debug=True)
