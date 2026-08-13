from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda:4" if torch.cuda.is_available() else "cpu"
 
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", dtype=torch.float16).to(device).eval()

raw_image = Image.open("../download/DW_open/12B/1601299731(1).jpg").convert('RGB')

question = "Question: what is in the image? Answer:"
inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device, torch.float16)
# print(inputs)

pixel_values = inputs["pixel_values"]

batch_size = pixel_values.shape[0]
image_embeds = model.vision_model(
    pixel_values,
    return_dict=True,
    interpolate_pos_encoding=False,
).last_hidden_state
image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
query_outputs = model.qformer(
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_attention_mask,
    return_dict=True,
)
query_output = query_outputs.last_hidden_state
language_model_inputs = model.language_projection(query_output)

print(language_model_inputs.shape)

outputs = model.generate(**inputs)
print(processor.batch_decode(outputs, skip_special_tokens=True)[0])