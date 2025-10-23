import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import joblib
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='templates')

model = joblib.load('./model/ml_pipeline.pkl')


@app.get('/', response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/predict')
def redirect_to_home():
    return RedirectResponse(url="/", status_code=303)


@app.post('/predict', response_class=HTMLResponse)
def prediction(
        request: Request,
        gender: str = Form(...),
        married: str = Form(...),
        applicant_income: float = Form(...),
        loan_amount: float = Form(...),
        credit_history: float = Form(...)
):
    gender_val = 1 if gender == 'Male' else 0
    married_val = 1 if married == 'Yes' else 0

    features = np.array([[gender_val, married_val, applicant_income, loan_amount, credit_history]])

    pred = model.predict(features)[0]
    print(f'Prediction: {pred}')
    prob = model.predict_proba(features)[0][1]

    status = 'Approved' if pred == 1 else 'Rejected'
    confidence = f"{prob * 100:.2f}%"

    return templates.TemplateResponse(
        'result.html',
        {
            'request': request,
            'status': status,
            'confidence': confidence,
            'form_data': {
                'gender': gender,
                'married': married,
                'applicant_income': applicant_income,
                'loan_amount': loan_amount,
                'credit_history': credit_history
            }
        }
    )


if __name__ == '__main__':
    uvicorn.run(app=app, host='localhost', port=8001)
