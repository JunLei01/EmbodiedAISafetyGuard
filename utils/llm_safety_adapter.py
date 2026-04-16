"""
MiniMax M2.7 LLM Safety Rule Generator
========================================

使用大语言模型动态生成安全规则，补充预定义模板的不足。

当场景实体不匹配现有 SafetyRuleTemplate 时，LLM 分析实体关系和属性，
识别潜在安全风险并生成对应的安全规则。

支持两种模式:
1. API 模式: 调用远程 MiniMax API
2. 本地模式: 使用本地部署的模型 (/models/MinMax-M2.7)
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch
import time
import hashlib

from utils.safety_knowledge_base import (
    SafetyRuleTemplate, SafetyCategory, Severity, Rule, Literal
)

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


@dataclass
class LLMGeneratedRule:
    """LLM 生成的安全规则"""
    rule_name: str
    hazard_predicate: str
    category: SafetyCategory
    severity: Severity
    probability: float
    natural_language: str
    prohibited_actions: List[str]
    involved_entities: List[str]
    reasoning: str  # LLM 的推理过程
    evidence: List[Dict[str, Any]]  # 支持证据


class MiniMaxLLMClient:
    """
    MiniMax M2.7 客户端

    支持两种模式:
    1. API 模式: 调用远程 API
    2. 本地模式: 使用本地部署的模型 (/models/)

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.minimaxi.chat/v1",
        model: str = "gpt-oss-20b",
        temperature: float = 0.3,
        max_tokens: int = 1000000,
        # 本地模式参数
        use_local: bool = False,
        local_model_path: str = "/models/gpt-oss-20b",
        device: str = "cuda",
    ):
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local = use_local
        self.local_model_path = local_model_path
        self.device = device

        # 本地模式初始化
        self.local_model = None
        self.local_tokenizer = None

        if self.use_local:
            self._init_local_model()
        elif not self.api_key:
            raise ValueError(
                "MiniMax API key required for API mode. "
                "Set MINIMAX_API_KEY env var or pass api_key, "
                "or set use_local=True for local deployment."
            )

        self._cache: Dict[str, Any] = {}  # 简单缓存

    def _init_local_model(self):
        """init local model and tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"Loading local MiniMax model from: {self.local_model_path}")

            self.local_tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path,
                trust_remote_code=True
            )

            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_path,
                # torch_dtype=torch.float16,
                # device_map="auto",
                trust_remote_code=True
            ).to(self.device)

            self.local_model.eval()
            print("Local LLM model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load local model: {e}")

    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()[:16]

    def analyze_scene_safety(
        self,
        entity_ids: List[str],
        grounded_facts: Dict[str, torch.Tensor],
        scene_name: Optional[str] = None,
    ) -> List[LLMGeneratedRule]:
        """
        Analy分析场景安全，识别预定义模板未覆盖的风险

        Args:
            entity_ids: 场景中的实体列表
            grounded_facts: 已推断的事实概率
            scene_name: 场景名称（可选）

        Returns:
            LLM 生成的安全规则列表
        """
        # 构建场景描述
        scene_description = self._build_scene_description(
            entity_ids, grounded_facts, scene_name
        )

        # 检查缓存
        cache_key = self._get_cache_key(scene_description)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 构建 Prompt
        prompt = self._build_safety_analysis_prompt(scene_description)

        try:
            response = self._call_llm(prompt)
            rules = self._parse_llm_response(response, entity_ids, grounded_facts)
            self._cache[cache_key] = rules
            return rules
        except Exception as e:
            print(f"  LLM 调用失败: {e}")
            return []

    def _build_scene_description(
        self,
        entity_ids: List[str],
        grounded_facts: Dict[str, torch.Tensor],
        scene_name: Optional[str] = None,
    ) -> str:
        """construct a natural language description of the scene for LLM input"""
        lines = ["Scene Analysis", "=" * 50]

        if scene_name:
            lines.append(f"Scene: {scene_name}")

        lines.append(f"\nEntities ({len(entity_ids)}):")
        for eid in entity_ids:
            # 提取实体相关的谓词
            entity_facts = [
                (k, v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in grounded_facts.items()
                if eid in k
            ]
            entity_facts.sort(key=lambda x: -x[1])  # 按概率降序

            lines.append(f"  - {eid}:")
            for fact, prob in entity_facts[:5]:  # 只显示前5个
                lines.append(f"      {fact}: {prob:.3f}")

        # 提取关系谓词
        lines.append("\nSpatial Relations:")
        relation_facts = [
            (k, v.item() if isinstance(v, torch.Tensor) else v)
            for k, v in grounded_facts.items()
            if any(r in k for r in ['near(', 'touching(', 'above(', 'inside('])
        ]
        relation_facts.sort(key=lambda x: -x[1])
        for fact, prob in relation_facts[:10]:
            lines.append(f"  {fact}: {prob:.3f}")

        return "\n".join(lines)

    def _build_safety_analysis_prompt(self, scene_description: str) -> str:
        return scene_description
    
    
# f"""You are a safety expert analyzing embodied AI scenes. Given the scene description below, identify potential safety hazards that are NOT covered by standard rules.

# {scene_description}

# Your task:
# 1. Analyze the entities and their relationships
# 2. Identify any unconventional or novel safety hazards
# 3. For each hazard, provide:
#    - Hazard name (snake_case)
#    - Category: FIRE, ELECTRICAL, CUT, BURN, CHEMICAL, FALL, CRUSH, CHILD_SAFETY, SLIP, POISON, COLLISION
#    - Severity: LOW, MEDIUM, HIGH, CRITICAL
#    - Probability (0.0-1.0)
#    - Description in natural language
#    - Affected entities
#    - Prohibited actions (e.g., "move_towards", "place_near", "pour", "touch")
#    - Reasoning

# Output format (JSON):
# {{
#   "hazards": [
#     {{
#       "name": "hazard_name",
#       "category": "FIRE",
#       "severity": "HIGH",
#       "probability": 0.85,
#       "description": "Natural language description",
#       "entities": ["Entity1", "Entity2"],
#       "prohibited_actions": ["action1", "action2"],
#       "reasoning": "Why this is a hazard"
#     }}
#   ]
# }}

# Focus on hazards involving unusual entity combinations. Only output valid JSON."""

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM（API 或本地模式）"""
        if self.use_local:
            return self._call_local_model(prompt)
        else:
            return self._call_api(prompt)

    def _call_local_model(self, prompt: str) -> str:
        """调用本地部署的模型"""
        try:
            import torch

            # # 构建对话格式
            # messages = [
            #     {"role": "system", "content": "You are a safety analysis expert."},
            #     {"role": "user", "content": prompt}
            # ]

            messages = [
                {"role": "user", "content": ROLE},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{\n  \"hazards\": [\n"}
            ]


            # 使用 tokenizer 构建输入
            inputs = self.local_tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)


            # with torch.no_grad():
            #     outputs = self.local_model.generate(
            #         **inputs,
            #         max_new_tokens=self.max_tokens,
            #         temperature=self.temperature,
            #         # do_sample=True,
            #         pad_token_id=self.local_tokenizer.eos_token_id
            #     )

            # response = self.local_tokenizer.decode(
            #     outputs[0][inputs.input_ids.shape[1]:],
            #     skip_special_tokens=True
            # )
            # print(f"====================Raw LLM Output:==========================\n{response}\n")
            response = """
            We need to ensure that we don't output any analysis. So we need to produce JSON with hazards array. Let's produce maybe 5 hazards to keep concise. But we can produce more. The instruction: "identify potential safety hazards that are NOT covered by standard rules." So we should list hazards that are not covered by standard rules. But we don't know what standard rules cover. But we can assume standard rules cover typical hazards like fire, electrical, cut, burn, chemical, fall, crush, child safety, slip, poison, collision. So we need hazards that are not covered by those categories. But we must still categorize them into one of those categories. So we need to find hazards that are not typical but still fit into categories. Maybe we can propose hazards like "candle near spray bottle" (chemical fire) is not typical but still fire/chemical. "hand towel holder near light switch" (electrical shock) is typical. But maybe "mirror near faucet" (glass break) is typical. But maybe "shower curtain near window" (slip) is typical. So maybe we need to find hazards that are not typical. But we can still produce them.

            But the instruction: "Identify any unconventional or novel safety hazards." So we need to identify hazards that are unconventional or novel. So we can produce hazards like "candle near spray bottle" (chemical fire), "hand towel holder near light switch" (electrical shock), "mirror near faucet" (glass break), "shower curtain near window" (slip). These are somewhat unconventional. But we can produce them.

            Let's produce 4 hazards: chemical fire, electrical shock, glass break, slip due to wet curtain.

            We need to provide name snake_case, category, severity, probability, description, entities, prohibited_actions, reasoning.

            Let's craft each:

            1. hazard_name: candle_near_spray_bottle
            category: CHEMICAL
            severity: HIGH
            probability: 0.6
            description: The candle flame is close to a spray bottle that may contain flammable cleaning chemicals, creating a risk of chemical fire if the flame contacts the spray.
            entities: ["Candle", "SprayBottle"]
            prohibited_actions: ["move_towards", "pour", "touch"]
            reasoning: The proximity of a flame to a potentially flammable liquid can ignite a chemical fire, which is not a standard fire hazard but a chemical fire hazard.

            2. hazard_name: hand_towel_holder_near_light_switch
            category: ELECTRICAL
            severity: HIGH
            probability: 0.4
            description: A wet hand towel held near a light switch can conduct electricity, posing a risk of electric shock if the switch is turned on.
            entities: ["HandTowelHolder", "LightSwitch"]
            prohibited_actions: ["move_towards", "touch"]
            reasoning: Wet towels near electrical switches can create conductive paths, leading to shock, which is a novel electrical hazard in this context.

            3. hazard_name: mirror_near_faucet
            category: CUT
            severity: MEDIUM
            probability: 0.3
            description: The fragile mirror is positioned close to the faucet; sudden water splashes or pressure changes could cause the mirror to break, producing glass shards that can cut occupants.
            entities: ["Mirror", "Faucet"]
            prohibited_actions: ["move_towards", "touch"]
            reasoning: The risk of glass breakage due to water pressure is a cut hazard that is not typically considered in standard rules.

            4. hazard_name: shower_curtain_near_window
            category: SLIP
            severity: LOW
            probability: 0.2
            description: The shower curtain is near the window; if it becomes wet, it can create a slippery surface on the floor near the window, increasing slip risk.
            entities: ["ShowerCurtain", "Window", "Floor"]
            prohibited_actions: ["move_towards"]
            reasoning: Wet curtains near windows can lead to slip hazards that are not commonly highlighted in standard rules.

            We can also add hazard for "garbage_can_near_toilet" but maybe not needed.

            We need to output JSON. Let's produce hazards array with these four hazards. Ensure valid JSON. Let's produce.assistantfinal{"hazards":[{"name":"candle_near_spray_bottle","category":"CHEMICAL","severity":"HIGH","probability":0.6,"description":"The candle flame is close to a spray bottle that may contain flammable cleaning chemicals, creating a risk of chemical fire if the flame contacts the spray.","entities":["Candle","SprayBottle"],"prohibited_actions":["move_towards","pour","touch"],"reasoning":"The proximity of a flame to a potentially flammable liquid can ignite a chemical fire, which is not a standard fire hazard but a chemical fire hazard."},{"name":"hand_towel_holder_near_light_switch","category":"ELECTRICAL","severity":"HIGH","probability":0.4,"description":"A wet hand towel held near a light switch can conduct electricity, posing a risk of electric shock if the switch is turned on.","entities":["HandTowelHolder","LightSwitch"],"prohibited_actions":["move_towards","touch"],"reasoning":"Wet towels near electrical switches can create conductive paths, leading to shock, which is a novel electrical hazard in this context."},{"name":"mirror_near_faucet","category":"CUT","severity":"MEDIUM","probability":0.3,"description":"The fragile mirror is positioned close to the faucet; sudden water splashes or pressure changes could cause the mirror to break, producing glass shards that can cut occupants.","entities":["Mirror","Faucet"],"prohibited_actions":["move_towards","touch"],"reasoning":"The risk of glass breakage due to water pressure is a cut hazard that is not typically considered in standard rules."},{"name":"shower_curtain_near_window","category":"SLIP","severity":"LOW","probability":0.2,"description":"The shower curtain is near the window; if it becomes wet, it can create a slippery surface on the floor near the window, increasing slip risk.","entities":["ShowerCurtain","Window","Floor"],"prohibited_actions":["move_towards"],"reasoning":"Wet curtains near windows can lead to slip hazards that are not commonly highlighted in standard rules."}]}

            """

            response = self.extract_hazards_json(response) 
            return response

        except Exception as e:
            raise RuntimeError(f"Local model inference failed: {e}")
        

    def _call_api(self, prompt: str) -> str:
        """调用 MiniMax API"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a safety analysis expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _parse_llm_response(
        self,
        response: str,
        entity_ids: List[str],
        grounded_facts: Dict[str, torch.Tensor],
    ) -> List[LLMGeneratedRule]:
        """解析 LLM 响应为结构化规则"""
        try:
            if isinstance(response, dict):
                data = response
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if not json_match:
                    return []
                data = json.loads(json_match.group())

            hazards = data.get("hazards", [])

            rules = []
            for hazard in hazards:
                try:
                    category = SafetyCategory(
                        hazard.get("category", "FIRE").lower()
                    )
                except:
                    category = SafetyCategory.FIRE

                try:
                    severity = Severity[
                        hazard.get("severity", "MEDIUM").upper()
                    ]
                except:
                    severity = Severity.MEDIUM

                rule = LLMGeneratedRule(
                    rule_name=f"llm_{hazard['name']}",
                    hazard_predicate=f"llm_hazard({','.join(hazard.get('entities', []))})",
                    category=category,
                    severity=severity,
                    probability=hazard.get("probability", 0.5),
                    natural_language=hazard.get("description", ""),
                    prohibited_actions=hazard.get("prohibited_actions", []),
                    involved_entities=hazard.get("entities", []),
                    reasoning=hazard.get("reasoning", ""),
                    evidence=[
                        {"source": "llm_analysis", "confidence": hazard.get("probability", 0.5)}
                    ],
                )
                rules.append(rule)

            return rules

        except Exception as e:
            print(f" LLM response parsing failed: {e}")
            return []
    
    def extract_hazards_json(self, response):
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


class LLMSafetyAdapter:
    """
    LLM 安全规则适配器

    将 LLM 生成的规则与符号推理系统集成
    """

    def __init__(
        self,
        llm_client: Optional[MiniMaxLLMClient] = None,
        enable_llm: bool = True,
        llm_threshold: float = 0.5,  # LLM 规则的最小概率阈值
    ):
        self.llm_client = llm_client
        self.enable_llm = enable_llm and (llm_client is not None)
        self.llm_threshold = llm_threshold

    def generate_rules(
        self,
        entity_ids: List[str],
        grounded_facts: Dict[str, torch.Tensor],
        activated_rules: List[Any],  # 已有的激活规则
        scene_name: Optional[str] = None,
    ) -> List[Any]:
        """
        生成补充规则

        当传统模板匹配生成的规则较少时，使用 LLM 补充
        """
        if not self.enable_llm or self.llm_client is None:
            return []

        # 如果已有足够规则，跳过 LLM 调用
        if len(activated_rules) >= 3:
            return []

        print(f"  LLM: 分析场景补充规则...")

        # 调用 LLM 生成规则
        llm_rules = self.llm_client.analyze_scene_safety(
            entity_ids, grounded_facts, scene_name
        )

        # 过滤低置信度规则
        filtered_rules = [
            r for r in llm_rules
            if r.probability >= self.llm_threshold
        ]

        # 转换为 ActivatedSafetyRule 格式
        converted_rules = self._convert_to_activated_rules(filtered_rules)

        print(f"  LLM: 生成 {len(converted_rules)} 条补充规则")

        return converted_rules

    def _convert_to_activated_rules(
        self, llm_rules: List[LLMGeneratedRule]
    ) -> List[Any]:
        """将 LLM 规则转换为 ActivatedSafetyRule 格式"""
        from utils.differentiable_safety_reasoner import (
            ActivatedSafetyRule, EvidenceItem
        )

        activated_rules = []
        for rule in llm_rules:
            # 构建 entity_bindings
            entity_bindings = {}
            for i, entity in enumerate(rule.involved_entities):
                entity_bindings[f"X{i}"] = entity

            # 构建证据
            evidence = [
                EvidenceItem(
                    predicate_key=e.get("source", "llm"),
                    probability=e.get("confidence", rule.probability),
                    source="llm_generated"
                )
                for e in rule.evidence
            ]

            activated_rule = ActivatedSafetyRule(
                rule_name=rule.rule_name,
                template_name="llm_generated",
                category=rule.category,
                severity=rule.severity,
                hazard_predicate=rule.hazard_predicate,
                probability=torch.tensor(rule.probability),
                entity_bindings=entity_bindings,
                evidence=evidence,
                proof_tree=None,  # LLM 规则无证明树
                prohibited_actions=rule.prohibited_actions,
                involved_entities=set(rule.involved_entities),
                mitigating_actions=[],
                natural_language=rule.natural_language,
                grounded_description=rule.reasoning,
            )
            activated_rules.append(activated_rule)

        return activated_rules
