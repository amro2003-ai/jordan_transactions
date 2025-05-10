from fastapi import FastAPI, Body, UploadFile, File
from pydantic import BaseModel
import uvicorn
import whisper
from rag import RAG

app = FastAPI()
LLM = RAG()
LLM.load_from_csv("jordan_transactions.csv")
model = whisper.load_model("base")

@app.post("/test/")
async def predict(x: str = Body(...)):
    answer = LLM.chat(x)
    print(answer)
    return {"answer": answer , "answer_ID" : 123}

@app.post("/audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()

    with open("temp_audio_file.wav", "wb") as f:
        f.write(contents)

    result = model.transcribe("temp_audio_file.wav")
    print(result)
    answer = LLM.chat(result["text"])

    return {"massege":answer}



uvicorn.run(app, host="0.0.0.0", port=8000)