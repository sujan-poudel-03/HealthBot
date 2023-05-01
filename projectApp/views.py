from urllib import response
from webbrowser import get
from django.shortcuts import render
from django.http import HttpResponse
from googletrans import Translator
from projectApp.models import chatbot
import nltk
from nltk.corpus import stopwords
import snowballstemmer

ne_stops = set(stopwords.words('nepali')) 

translator = Translator()
translator = Translator(service_urls=['translate.googleapis.com'])

class Stem:
    """Stem the words to its root eg 'गरेका' to 'गर'.
    Credit: https://github.com/snowballstem/snowball
    """
    def __init__(self) -> None:
        self.stemmer = snowballstemmer.NepaliStemmer()
    
    def rootify(self, text):
        """Generates the stem words for input text.

        Args:
            text (Union(List, str)): Text to be stemmed or lemmatized.

        Returns:
            Union(List, str): stemmed text.
        """
        if isinstance(text, str):
            return self.stemmer.stemWords(text.split())
        
        return self.stemmer.stemWords(text)

messages = []

import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)

if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

def sentenceInput(sent):
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []
  batch_size = 32
  encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

  # Add the encoded sentence to the list.    
  input_ids.append(encoded_dict['input_ids'])
      
  # And its attention mask (simply differentiates padding from non-padding).
  attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)

  # print('Original :', sent)
  # print('Token IDs :', input_ids)
  # print('Attention Masks :', attention_masks)

  # Create the DataLoader.
  prediction_data = TensorDataset(input_ids, attention_masks)
  prediction_sampler = SequentialSampler(prediction_data)
  prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

#   print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

  # Put model in evaluation mode
  model1 = torch.load('mBERT.pth', map_location=map_location)
  model1.eval()

  # Tracking variables 
  predictions = []

  # Predict 
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(map_location) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        result = model1(b_input_ids, 
                      token_type_ids=None, 
                      attention_mask=b_input_mask,
                      return_dict=True)

    logits = result.logits

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    
    # Store predictions and true labels
    predictions.append(logits)

  # print(predictions)
  list_of_intent = ['अभिवादन', 'सकरात्मक', 'नकरात्मक', 'अलविदा', 'रोग र प्रकार', 'एलर्जी परिभाषा वा जानकारी', 'एलर्जीको लक्ष्ण', 'एलर्जीको कारण', 'एलर्जीको परिक्षण वा जाँच', 'एलर्जीको उपचार', 'एलर्जीको रोकथाम', 'निमोनिया परिभाषा वा जानकारी', 'निमोनियाको लक्ष्ण', 'निमोनियाको कारण', 'निमोनियाको परिक्षण वा जाँच', 'निमोनियाको उपचार', 'निमोनियाको रोकथाम', 'कोभिड-१९ परिभाषा वा जानकारी', 'कोभिड-१९को लक्ष्ण', 'कोभिड-१९को कारण', 'कोभिड-१९को परिक्षण वा जाँच', 'कोभिड-१९को उपचार', 'कोभिड-१९को रोकथाम', 'मधुमेह परिभाषा वा जानकारी', 'मधुमेहको लक्ष्ण', 'मधुमेहको कारण', 'मधुमेहको परिक्षण वा जाँच', 'मधुमेहको उपचार', 'मधुमेहको रोकथाम', 'सामान्य चिसो परिभाषा वा जानकारी', 'सामान्य चिसोको लक्ष्ण', 'सामान्य चिसोको कारण', 'सामान्य चिसोको परिक्षण वा जाँच', 'सामान्य चिसोको उपचार', 'सामान्य चिसोको रोकथाम', 'क्षयरोग परिभाषा वा जानकारी', 'क्षयरोगको लक्ष्ण', 'क्षयरोगको कारण', 'क्षयरोगको परिक्षण वा जाँच', 'क्षयरोगको उपचार', 'क्षयरोगको रोकथाम', 'ग्यास्ट्रिक परिभाषा वा जानकारी', 'ग्यास्ट्रिकको लक्ष्ण', 'ग्यास्ट्रिकको कारण', 'ग्यास्ट्रिकको परिक्षण वा जाँच', 'ग्यास्ट्रिकको उपचार', 'ग्यास्ट्रिकको रोकथाम', 'हृदयघात परिभाषा वा जानकारी', 'हृदयघातको लक्ष्ण', 'हृदयघातको कारण', 'हृदयघातको परिक्षण वा जाँच', 'हृदयघातको उपचार', 'हृदयघातको रोकथाम']
  # print(list_of_intent[1])
  prediction = predictions[0]
  pred = prediction[0]
  prediction_output = {  }
  prediction_output = dict(zip(list_of_intent, pred))
#   print(prediction_output)

  fin_max = max(prediction_output, key=prediction_output.get)
#   print("Intent of the user:",fin_max)
  return fin_max

dataframe = pd.read_excel('responses.xlsx')
tag = dataframe.Intent_Category.values
reply = dataframe.Responses.values

result = ''

def get_response(message): 
  intent = sentenceInput(message)
  print(intent)
  for i in range(53):
    if tag[i] == intent:
      result = reply[i]
      break
  return result

# get_response(sent)

def text_check(textInput):
    flag = 0
    data_into_list = textInput.split(" ")
    print(data_into_list)
    nepaliStemmer = Stem()
    stemText = nepaliStemmer.rootify(textInput)
    print(stemText)
    length = len(stemText)
    # print(length)
    
    with open('text.txt', encoding='utf-8') as f:
        file_text = f.read()
        file_text_list = file_text.split("\n")
        # print(file_text_list[10])
        #print(file_text_list)
        
        for sText in stemText:
            #print(sText)
            if sText in file_text_list:
                # print(sText)
                flag = 1
    return flag

# Create your views here.
def index(request):
    inp = None
    if request.method == 'POST':
        inp = request.POST.get('textinput')
        detectedLang = translator.detect(inp) 
        # print(detectedLang.lang)
        # print(detectedLang)
        if detectedLang.lang == 'en':
            modelInput = []
            print('Input is in English language')
            englishToNepali = translator.translate(inp, dest='ne', src='en')
            nepali_Text = englishToNepali.text
            print('Text =', nepali_Text)
            # # tokenized = nltk.word_tokenize(nepaliText)
            # # print('tokenized = ', tokenized)
            # # for word in tokenized: 
            # #     if word not in ne_stops:
            # #         # print("word = ",word)
            # #         modelInput.append(word)
            
            # # nepaliStemmer = Stem()
            # # sent = nepaliStemmer.rootify(modelInput)
            result = ''
            print(text_check(nepali_Text))
            if text_check(nepali_Text) == 1:
                # output = get_response(nepali_Text)
                intent = sentenceInput(nepali_Text)
                for i in range(53):
                    if tag[i] == intent:
                        result = reply[i]
                        break            
            else:
                result = "माफ गर्नुहोस्, म तपाईंको सन्देश बुझ्न असमर्थ छु।"
                
            return render(request, 'index.html', {'messages': inp, 'output':result})                    
            
        elif detectedLang.lang == 'ne':
            modelInput = []
            inputText = inp
            print('Input is in Nepali language')
            # nepaliToEnglish = translator.translate(inp, dest='en', src='ne')
            # englishText = nepaliToEnglish.text
            # print(englishText)
            # print(inp)
            # tokenized = nltk.word_tokenize(inp)
            # # print(tokenized)
            # for word in tokenized: 
            #     if word not in ne_stops:
            #         # print(word)
            #         modelInput.append(word)

            # nepaliStemmer = Stem()
            # sent = nepaliStemmer.rootify(modelInput)
            # print(sent)
            result = ''
            print(text_check(inputText))
            if text_check(inputText) == 1:
                intent = sentenceInput(inputText)
                for i in range(53):
                    if tag[i] == intent:
                        result = reply[i]
                        break
            else:
                result = "माफ गर्नुहोस्, म तपाईंको सन्देश बुझ्न असमर्थ छु।"
                
            return render(request, 'index.html', {'messages': inp, 'output':result})                    
                               
        else:
            modelInput = []
            print('Input is in Roman Nepali language')
            romanToNepali = translator.translate(inp, dest='ne', src='en')
            nepaliText = romanToNepali.text
            # print('-------------', nepaliText)
            tokenized = nltk.word_tokenize(nepaliText)
            # print(tokenized)
            # for word in tokenized: 
            #     if word not in ne_stops:
            #         # print(word)
            #         modelInput.append(word)
                    
            # nepaliStemmer = Stem()
            # sent = nepaliStemmer.rootify(modelInput)
            print(nepaliText)
            result = ''
            print(text_check(nepaliText))
            if text_check(nepaliText) == 1:
                intent = sentenceInput(nepaliText)
                for i in range(53):
                    if tag[i] == intent:
                        result = reply[i]
                        break
            else:
                result = "माफ गर्नुहोस्, म तपाईंको सन्देश बुझ्न असमर्थ छु।"                
            
            return render(request, 'index.html', {'messages': inp, 'output':result})                    
    
    return render(request, 'index.html')                    


from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.decorators import APIView
from .serializers import chatbotSerializer
from .models import chatbot
from django.http import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status
import pandas as pd

def chat_list(request):
    # get all chat
    if request.method == 'GET':
        chat = chatbot.objects.all().order_by('-id')[:1]
        serializer = chatbotSerializer(chat, many=True)
        return JsonResponse(serializer.data, safe=False)
    
    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = chatbotSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)

# @api_view(['GET','POST'])
# def react(request):
    
#     if request.method == 'GET':
#         chat = chatbot.objects.all().order_by('-id')[:1]
#         serializer = chatbotSerializer(chat, many=True)
#         return Response(serializer.data)

#     elif request.method == 'POST': 
#         serializer = chatbotSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class reactAPI(APIView):
    def get(self, request):
        output = pd.read_excel('./responses.xlsx')
        chat = chatbot.objects.all().order_by('-id')[:1]
        serializer = chatbotSerializer(chat, many=True)
        return Response(serializer.data)

    def post(self, request):
        data = request.data
        print('Message : ', data['userInput'])
        pred_intent = " "
        pred_response = " "
        new_data = chatbot.objects.create(userInput=data['userInput'], intentClassified=pred_intent, response=pred_response)
        serializer = chatbotSerializer(data=new_data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
