import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)


sent = "एलर्राेजी को लक्ष्ण"
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

  print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

  # Put model in evaluation mode
  model1 = torch.load('mBERT.pth')
  model1.eval()

  # Tracking variables 
  predictions = []

  # Predict 
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to("cuda") for t in batch)
    
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
  list_of_intent = ['अभिवादन', 'सकरात्मक', 'नकरात्मक', 'अलविदा', 'रोग र प्रकार', 'एलर्जी परिभाषा वा जानकारी', 'एलर्जीको लक्ष्ण', 'एलर्जीको कारण', 'एलर्जीको परिक्षण वा जाँच', 'एलर्जीको उपचार', 'एलर्जीको रोकथाम', ' निमोनिया परिभाषा वा जानकारी', ' निमोनियाको लक्ष्ण', ' निमोनियाको कारण', ' निमोनियाको परिक्षण वा जाँच', ' निमोनियाको उपचार', ' निमोनियाको रोकथाम', ' कोभिड-१९ परिभाषा वा जानकारी', ' कोभिड-१९को लक्ष्ण', ' कोभिड-१९को कारण', ' कोभिड-१९को परिक्षण वा जाँच', ' कोभिड-१९को उपचार', ' कोभिड-१९को रोकथाम', 'मधुमेह परिभाषा वा जानकारी', 'मधुमेहको लक्ष्ण', 'मधुमेहको कारण', 'मधुमेहको परिक्षण वा जाँच', 'मधुमेहको उपचार', 'मधुमेहको रोकथाम', 'सामान्य चिसो परिभाषा वा जानकारी', 'सामान्य चिसोको लक्ष्ण', 'सामान्य चिसोको कारण', 'सामान्य चिसोको परिक्षण वा जाँच', 'सामान्य चिसोको उपचार', 'सामान्य चिसोको रोकथाम', 'क्षयरोग परिभाषा वा जानकारी', 'क्षयरोगको लक्ष्ण', 'क्षयरोगको कारण', 'क्षयरोगको परिक्षण वा जाँच', 'क्षयरोगको उपचार', 'क्षयरोगको रोकथाम', 'ग्यास्ट्रिक परिभाषा वा जानकारी', 'ग्यास्ट्रिकको लक्ष्ण', 'ग्यास्ट्रिकको कारण', 'ग्यास्ट्रिकको परिक्षण वा जाँच', 'ग्यास्ट्रिकको उपचार', 'ग्यास्ट्रिकको रोकथाम', 'हृदयघात परिभाषा वा जानकारी', 'हृदयघातको लक्ष्ण', 'हृदयघातको कारण', 'हृदयघातको परिक्षण वा जाँच', 'हृदयघातको उपचार', 'हृदयघातको रोकथाम']
  # print(list_of_intent[1])
  prediction = predictions[0]
  pred = prediction[0]
  prediction_output = {  }
  prediction_output = dict(zip(list_of_intent, pred))
  print(prediction_output)

  fin_max = max(prediction_output, key=prediction_output.get)
  print("Intent of the user:",fin_max)
  return fin_max

dataframe = pd.read_excel('responses.xlsx')
tag = dataframe.Intent_Category.values
reply = dataframe.Responses.values

def get_response(message): 
  intent = sentenceInput(sent)
  for i in range(53):
    if tag[i] == intent:
      result = reply[i]
      break
  # print(f"Response : {result}")
  return "Intent: "+ intent + ' : ' + "Response: " + result

get_response(sent)