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
    strCallInText = '''I hope this message finds you amidst your day. I'm reaching out with a heavy heart to share some sad news. There's been a loss in our family, and I wanted to let you know as soon as possible.

We've lost  and it's been incredibly tough for all of us. The void they've left behind is immense, and we're grappling with a mix of emotions.

I know this news might come as a shock, and I understand if you need some time to process it. But if you can spare a moment, your support and presence would mean the world to us. Even a quick call or a message to offer your condolences would be deeply appreciated.

We're planning , and your presence would bring some comfort during this difficult time. If you can make it, please let me know as soon as possible so we can coordinate accordingly.

In the meantime, if you need anything or just want to talk, I'm here for you.

Sending you love and strength during this challenging time.'''

    # print(TextProcessing(strCallInText=strCallInText))


    print(HandleCallInput(r"transtest2.mp3"))

