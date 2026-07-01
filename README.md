# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions via dialogue.
See DEPLOY_AND_SUBMIT.md for deployment steps and APPROACH.md for the design write-up.

## Local run
```
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --reload
```
Then visit http://localhost:8000/docs

## Structure
- app/          FastAPI service, agent logic, retrieval, prompts
- data/         cleaned catalog (370 Individual Test Solutions)
- tests/        offline pipeline tests + trace eval harness
- traces/       provided sample conversations (for eval)
