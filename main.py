from fastapi import FastAPI
import joblib
import uvicorn

app = FastAPI()
model = joblib.load("models/project_classifier.pkl")

@app.get("/")
def read_root():
    return {"message": "🚀 AI Model for Project Management is Running"}

@app.post("/predict_project_status/")
def predict_status(budget: float, deadline_days: int, risk_level: int):
    prediction = model.predict([[budget, deadline_days, risk_level]])
    status_map = {0: "Pending", 1: "In Progress", 2: "Completed"}
    return {"Predicted_Status": status_map[prediction[0]]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
