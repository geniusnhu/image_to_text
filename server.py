from fastapi import FastAPI,Request,File,UploadFile,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os, io
import pandas as pd
import numpy as np
from torch_utils import get_classification



app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")

# Load model and embeddings
def get_image(x):
  return x.split(', ')[0]

ALLOWED_EXTENSION = {'jpg', 'png', 'pdf', 'tif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/extract_text")
async def predict(image: UploadFile = File(...)):
    if not allowed_file(image.filename):
        return {"filename": image.filename, 'error': 'format is not supported ! ! !'}

    #contents = _save_file_to_disk(image, path="temp", save_as="temp")
    contents = image.file
    result = await get_classification(contents, image.filename)
    #response = StreamingResponse(
    #    io.StringIO(result.to_csv(index=False)),
    #    media_type="text/csv",
    #    headers={'Content-Disposition': 'filename=result.csv'}
    #)
    #response = FileResponse(
    #    path='images/result.csv', media_type='application/octet-stream',
    #    filename='result.csv')
    #return response #{"filename": image.filename, "result": response}
    #return response

@app.get("/extract_text/download") #/download")
async def download_file():

    response = FileResponse(
        path='images/result.csv', media_type='application/octet-stream',
        filename='result.csv')

    return {'output': response}

@app.post('/download')
def form_post(request: Request, action: str = Form(...)):
    if action == 'download':
        # Requires aiofiles
        result = FileResponse(
            path='images/result.csv', media_type='application/octet-stream',
            filename='result.csv')
        return result

def _save_file_to_disk(uploaded_file, path=".", save_as="default"):
    extension = os.path.splitext(uploaded_file.filename)[-1]
    temp_file = os.path.join(path, save_as + extension)
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file


if __name__ == '__main__':
    uvicorn.run(app=app, port=5000, log_level="info")
