# from transformers import pipeline

# pipe = pipeline("text-generation", model="/data/junlei/NPG/models/MiniMax-2.7", trust_remote_code=True)
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe(messages)


# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_path = "/data/junlei/NPG/models/MiniMax-2.7/"

# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU device: {torch.cuda.get_device_name(0)}")

# device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True)
# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# inputs = tokenizer.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=256,
#     do_sample=True,
#     temperature=1.0,
#     top_p=0.95,
#     top_k=40,
# )

# print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText

# model_path = "/data/junlei/NPG/models/Qwen3.5-35B/"
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU device: {torch.cuda.get_device_name(0)}")

# device = "cuda" if torch.cuda.is_available() else "cpu"


# processor = AutoProcessor.from_pretrained(model_path)
# model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
# messages = [
#     {
#         "role": "user",
#         "content": [
#             # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "Who are you?"}
#         ]
#     },
# ]
# inputs = processor.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)


# outputs = model.generate(
#     **inputs,
#     max_new_tokens=40,
#     do_sample=True,
#     temperature=0.7,     
#     top_p=0.9,
#     renormalize_logits=True,
#     output_scores=False,
# )

# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CAUSAL_CONV1D_FORCE_CPU_IMPL"] = "1"
# import torch
# from transformers import AutoProcessor, AutoModelForImageTextToText

# model_path = "/data/junlei/NPG/models/Qwen3.5-9B/"

# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU device: {torch.cuda.get_device_name(0)}")

# processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     device_map="auto",
#     dtype=torch.bfloat16,          # 使用 float32
#     trust_remote_code=True
# )
# print("Model vocab size:", model.config.vocab_size)
# print("Tokenizer vocab size:", len(processor.tokenizer))

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "Who are you?"}
#         ]
#     },
# ]
# inputs = processor.apply_chat_template(
# 	messages,
# 	add_generation_prompt=True,
# 	tokenize=True,
# 	return_dict=True,
# 	return_tensors="pt",
# ).to(model.device)

# print("input_ids shape:", inputs['input_ids'].shape)
# print("input_ids min/max:", inputs['input_ids'].min().item(), inputs['input_ids'].max().item())
# print("attention_mask sum per sample:", inputs['attention_mask'].sum(dim=1))

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=40,
#     do_sample=True,
#     temperature=0.7,     
#     top_p=0.9,
#     renormalize_logits=True,
#     output_scores=False,
# )

# print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))


from transformers import pipeline
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/data/junlei/NPG/models/gpt-oss-20b"
# model_path = "/data/junlei/NPG/models/Qwen3.5-9B/"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")

# 检查 vocab 是否匹配
# print(f"Model vocab size: {model.config.vocab_size}")
# print(f"Tokenizer vocab size: {len(tokenizer)}")

# 检查 input_ids 是否越界
# 假设你的 inputs 变量是这样构造的
# inputs = tokenizer("your text", return_tensors="pt").to("cuda")
# print(f"Input IDs max: {inputs['input_ids'].max().item()}")
# print(f"Input IDs min: {inputs['input_ids'].min().item()}")

# pipe = pipeline(
#     "text-generation",
#     model=model_id,
#     torch_dtype="auto",
#     device_map="auto",
# )

SCENE_DES = """ \n\nScene Analysis\n==================================================\n\nEntities (30):\n  - Bathtub:\n      near(BathtubBasin,Cloth): 0.929\n      near(Cloth,BathtubBasin): 0.929\n      near(Bathtub,BathtubBasin): 0.804\n      near(BathtubBasin,Bathtub): 0.804\n      near(Bathtub,Cloth): 0.783\n  - BathtubBasin:\n      near(BathtubBasin,Cloth): 0.929\n      near(Cloth,BathtubBasin): 0.929\n      near(Bathtub,BathtubBasin): 0.804\n      near(BathtubBasin,Bathtub): 0.804\n      above(Bathtub,BathtubBasin): 0.496\n  - Candle:\n      flammable(Candle): 0.950\n      above(DishSponge,Candle): 0.837\n      near(Candle,SideTable): 0.810\n      near(SideTable,Candle): 0.810\n      near(Candle,SprayBottle): 0.784\n  - Cloth:\n      near(BathtubBasin,Cloth): 0.929\n      near(Cloth,BathtubBasin): 0.929\n      near(Bathtub,Cloth): 0.783\n      near(Cloth,Bathtub): 0.783\n      above(Bathtub,Cloth): 0.473\n  - DishSponge:\n      near(DishSponge,PaperTowelRoll): 0.847\n      near(PaperTowelRoll,DishSponge): 0.847\n      above(DishSponge,Candle): 0.837\n      above(DishSponge,SideTable): 0.813\n      above(DishSponge,SprayBottle): 0.502\n  - Faucet:\n      near(Faucet,SoapBottle): 0.891\n      near(SoapBottle,Faucet): 0.891\n      above(Mirror,Faucet): 0.859\n      near(Faucet,SinkBasin): 0.795\n      near(SinkBasin,Faucet): 0.795\n  - Floor:\n      near(Floor,SoapBar): 0.016\n      near(SoapBar,Floor): 0.016\n      above(SoapBar,Floor): 0.009\n      near(Floor,ToiletPaperHanger): 0.001\n      near(ToiletPaperHanger,Floor): 0.001\n  - GarbageCan:\n      near(GarbageCan,Toilet): 0.095\n      near(Toilet,GarbageCan): 0.095\n      near(GarbageCan,ScrubBrush): 0.003\n      near(ScrubBrush,GarbageCan): 0.003\n      near(GarbageCan,Plunger): 0.001\n  - HandTowel:\n      near(HandTowel,HandTowelHolder): 0.952\n      near(HandTowelHolder,HandTowel): 0.952\n      near(HandTowelHolder,LightSwitch): 0.692\n      near(LightSwitch,HandTowelHolder): 0.692\n      near(HandTowel,LightSwitch): 0.662\n  - HandTowelHolder:\n      near(HandTowel,HandTowelHolder): 0.952\n      near(HandTowelHolder,HandTowel): 0.952\n      near(HandTowelHolder,LightSwitch): 0.692\n      near(LightSwitch,HandTowelHolder): 0.692\n      touching(HandTowel,HandTowelHolder): 0.615\n  - LightSwitch:\n      near(HandTowelHolder,LightSwitch): 0.692\n      near(LightSwitch,HandTowelHolder): 0.692\n      near(HandTowel,LightSwitch): 0.662\n      near(LightSwitch,HandTowel): 0.662\n      above(LightSwitch,HandTowel): 0.031\n  - Mirror:\n      above(Mirror,Faucet): 0.859\n      fragile(Mirror): 0.850\n      above(Mirror,SoapBottle): 0.623\n      above(Mirror,Sink): 0.420\n      near(Faucet,Mirror): 0.384\n  - PaperTowelRoll:\n      above(PaperTowelRoll,SprayBottle): 0.908\n      above(PaperTowelRoll,SideTable): 0.863\n      near(DishSponge,PaperTowelRoll): 0.847\n      near(PaperTowelRoll,DishSponge): 0.847\n      above(PaperTowelRoll,Candle): 0.421\n  - Plunger:\n      near(Plunger,ScrubBrush): 0.898\n      near(ScrubBrush,Plunger): 0.898\n      near(Plunger,Toilet): 0.172\n      near(Toilet,Plunger): 0.172\n      touching(Plunger,ScrubBrush): 0.172\n  - ScrubBrush:\n      near(Plunger,ScrubBrush): 0.898\n      near(ScrubBrush,Plunger): 0.898\n      near(ScrubBrush,Toilet): 0.453\n      near(Toilet,ScrubBrush): 0.453\n      touching(Plunger,ScrubBrush): 0.172\n  - Shelf:\n      above(Shelf,ToiletPaper): 0.481\n      near(Shelf,ToiletPaper): 0.333\n      near(ToiletPaper,Shelf): 0.333\n      touching(Shelf,ToiletPaper): 0.000\n      touching(ToiletPaper,Shelf): 0.000\n  - ShowerCurtain:\n      above(ShowerCurtain,Towel): 0.100\n      above(ShowerCurtain,Window): 0.013\n      near(ShowerCurtain,Window): 0.005\n      near(Window,ShowerCurtain): 0.005\n      near(ShowerCurtain,Towel): 0.002\n  - ShowerHead:\n      near(ShowerHead,ToiletPaperHanger): 0.698\n      near(ToiletPaperHanger,ShowerHead): 0.698\n      near(ShowerHead,SoapBar): 0.176\n      near(SoapBar,ShowerHead): 0.176\n      above(ShowerHead,Plunger): 0.081\n  - SideTable:\n      near(SideTable,SprayBottle): 0.863\n      near(SprayBottle,SideTable): 0.863\n      above(PaperTowelRoll,SideTable): 0.863\n      above(DishSponge,SideTable): 0.813\n      near(Candle,SideTable): 0.810\n  - Sink:\n      above(SinkBasin,Sink): 0.932\n      near(Faucet,SinkBasin): 0.795\n      near(SinkBasin,Faucet): 0.795\n      near(SinkBasin,SoapBottle): 0.664\n      near(SoapBottle,SinkBasin): 0.664\n  - SinkBasin:\n      above(SinkBasin,Sink): 0.932\n      near(Faucet,SinkBasin): 0.795\n      near(SinkBasin,Faucet): 0.795\n      near(SinkBasin,SoapBottle): 0.664\n      near(SoapBottle,SinkBasin): 0.664\n  - SoapBar:\n      near(SoapBar,ToiletPaperHanger): 0.653\n      near(ToiletPaperHanger,SoapBar): 0.653\n      near(ShowerHead,SoapBar): 0.176\n  """

ROLE = """You are a safety expert analyzing embodied AI scenes. Given the scene description below, identify potential safety hazards that are NOT covered by standard rules.
Your task:
1. Analyze the entities and their relationships
2. Identify any unconventional or novel safety hazards
3. For each hazard, provide:
   - Hazard name (snake_case)
   - Category: FIRE, ELECTRICAL, CUT, BURN, CHEMICAL, FALL, CRUSH, CHILD_SAFETY, SLIP, POISON, COLLISION
   - Severity: LOW, MEDIUM, HIGH, CRITICAL
   - Probability (0.0-1.0)
   - Description in natural language
   - Affected entities
   - Prohibited actions (e.g., "move_towards", "place_near", "pour", "touch")
   - Reasoning

Output format (JSON):
{{
  "hazards": [
    {{
      "name": "hazard_name",
      "category": "FIRE",
      "severity": "HIGH",
      "probability": 0.85,
      "description": "Natural language description",
      "entities": ["Entity1", "Entity2"],
      "prohibited_actions": ["action1", "action2"],
      "reasoning": "Why this is a hazard"
    }}
  ]
}}

CRITICAL INSTRUCTION: 
Do NOT output any thinking process, analysis, or explanations. 
You must start your response EXACTLY with the character '{' and end with '}'. 
Return ONLY valid JSON.
"""

messages = [
    {"role": "user", "content": ROLE},
    {"role": "user", "content": SCENE_DES},
    {"role": "assistant", "content": "{\n  \"hazards\": [\n"}
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

def extract_hazards_json(response):
    """
    Get and parse JSON data from the model's output with specific tags
    """
    json_str = ""
    
    start_tag = "<|channel|>final<|message|>"
    end_tag = "<|return|>"
    
    if start_tag in response:
        temp_str = response.split(start_tag)[1]
        if end_tag in temp_str:
            json_str = temp_str.split(end_tag)[0].strip()
        else:
            json_str = temp_str.strip()
            
    if not json_str:
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            
    if json_str:
        try:
            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON failed to parse: {e}")
            print(f"Extracted string:\n{json_str}")
            return None
    else:
        print("Cannot find valid JSON structure in the output.")
        return None


outputs = model.generate(**inputs, max_new_tokens=100000000)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])

json_result = extract_hazards_json(response)
print(json_result)
