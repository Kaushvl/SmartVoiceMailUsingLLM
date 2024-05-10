from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import dotenv
import os
import json
import pymongo
dotenv.load_dotenv()
groq_api_key = os.environ.get('groq_api_key')
from dataclasses import dataclass
from faster_whisper import WhisperModel
from datetime import datetime

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['CallSum']
Collection = db['CallData']

def SaveToDB(jsonCallData):
    jsonCallData["DateTime"] = datetime.now()
    Collection.insert_one(jsonCallData)
    print(("Data Uploaded to database"))


def TextProcessing(strCallInText):
    chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    system = '''You need to perform this task with vary high accuracy. Task : I am going to provide you a text, which is originally a call message from a person, you need to analize urgency of call, Give a title based on Call text and  summarize this less than in 30 words and only return the output in json without any description fromat as 'CallUrgency':'YourAnswerForUrgency','CallTitle':'YourAnswerForCallTitle','CallSummary':'YourAnswerForUrgency' '''
    human = "{strCallInText}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    strResponse = chain.invoke({strCallInText}).content
    jsonResponse = json.loads(strResponse)
    jsonCallData = jsonResponse
    jsonCallData['strCallInText'] = strCallInText
    SaveToDB(jsonCallData)

    return jsonResponse

def TranscribeAudioFile(strFilePath):
    model_size = "large-v3"
    # Run on GPU with FP16
    model = WhisperModel(model_size, device="cuda", compute_type="int8",cpu_threads=8)
    # or run on GPU with INT8
    # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    # or run on CPU with INT8
    # model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(strFilePath, beam_size=1)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    strAudioContent = ""
    for segment in segments:
        strAudioContent += segment.text
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    return strAudioContent

def HandleCallInput(strAudioFilePath):
    strAudioContent = TranscribeAudioFile(strAudioFilePath)
    jsonResponse = TextProcessing(strAudioContent)
    return jsonResponse



if __name__ == '__main__':

    print(HandleCallInput(r"Data\transtest1.mp3"))

