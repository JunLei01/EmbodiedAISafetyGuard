"""
LLaVA-3D Scene Recognizer Integration
======================================
Wrapper for LLaVA-3D to integrate with the neuro-symbolic safety pipeline.

This module provides:
    1. LLaVA3DRecognizer: Wrapper class for running LLaVA-3D inference
    2. Prompt templates for structured output parsing
    3. Output parsing to Prolog-style predicates
"""

import os
import sys
import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass



def add_llava_to_path():
    """Add LLaVA-3D to Python path if not already there."""
    llava_path = os.path.join(os.path.dirname(__file__), 'LLaVA-3D')
    if llava_path not in sys.path:
        sys.path.insert(0, llava_path)


add_llava_to_path()

try:
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import (
        process_images,
        process_videos,
        tokenizer_special_token,
        get_model_name_from_path,
    )
    from PIL import Image
    LLAVA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LLaVA-3D imports failed: {e}")
    LLAVA_AVAILABLE = False


@dataclass
class SceneRecognitionResult:
    """Result from LLaVA-3D scene recognition."""
    raw_output: str                           # Original LLaVA-3D text output
    predicates: str                           # Parsed Prolog-style predicates
    entities: List[Dict]                      # Structured entity list
    image_path: Optional[str] = None          # Source image/video path
    query: str = ""                           # Query used for recognition


# Prompt template for generating Prolog-style predicates
SCENE_UNDERSTANDING_PROMPT = """You are a scene understanding AI for robot safety. Analyze the image and identify:

1. All objects/entities with their types (flammable, sharp, electrical, etc.)
2. Spatial relationships between objects (near, on, above, etc.)
3. 3D positions if visible (x, y, z coordinates in meters)
4. Relevant attributes (temperature, state, etc.)

Output in the following Prolog-style format (one predicate per line):

object(id, category, [type1, type2, ...]).
position(id, x, y, z).
attribute(id, name, value).
relation(id1, rel_type, id2).

Example output:
object(1, gasoline_can, [flammable, container]).
object(2, stove, [fire_source, hot]).
position(1, 1.2, 0.5, 0.8).
position(2, 1.0, 0.5, 0.8).
attribute(2, temperature, 200).
near(1, 2).

Describe this scene for robot safety assessment:"""


SAFETY_FOCUS_PROMPT = """You are a safety-aware scene analyzer for embodied robotics.
Identify ALL potential safety hazards in this scene including:
- Fire hazards (flammable objects + heat sources)
- Electrical hazards (water near electronics)
- Physical hazards (sharp objects, heavy objects, unstable stacks)
- Human safety (children near dangers, elderly persons)

Output as Prolog predicates (one per line):
object(id, category, [semantic_types]).
position(id, x, y, z).
attribute(id, attr_name, value).
relation(id1, relation, id2).

Scene to analyze:"""


class LLaVA3DRecognizer:
    """
    Wrapper for LLaVA-3D model to perform scene recognition.

    Usage:
        recognizer = LLaVA3DRecognizer(
            model_path="path/to/llava-3d-model",
            precision="bf16"
        )

        result = recognizer.recognize_image(
            image_path="kitchen.jpg",
            prompt=SCENE_UNDERSTANDING_PROMPT
        )

        # Parse to predicates
        grounded_facts, entity_ids = recognizer.parse_to_predicates(result)
    """

    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        precision: str = "bf16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize LLaVA-3D model.

        Args:
            model_path: Path to pretrained LLaVA-3D model
            model_base: Base model path (for delta weights)
            precision: "fp32", "bf16", or "fp16"
            device: Device to run on
        """
        if device.startswith("cuda"):
            import os 
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[0]  # Use only the first GPU
            device = "cuda"

        if not LLAVA_AVAILABLE:
            raise RuntimeError("LLaVA-3D dependencies not available. "
                             "Make sure LLaVA-3D is cloned and dependencies are installed.")

        self.model_path = model_path
        self.model_base = model_base
        self.precision = precision
        self.device = device

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the LLaVA-3D model."""
        disable_torch_init()

        # Set dtype
        if self.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        elif self.precision == "fp16":
            self.torch_dtype = torch.half
        else:
            self.torch_dtype = torch.float32

        # Determine conv mode
        model_name = get_model_name_from_path(self.model_path)
        if "3D" in model_name:
            self.conv_mode = "llava_v1"
        elif "v1" in model_name.lower():
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v0"

        print(f"Loading model from {self.model_path}...")
        self.tokenizer, self.model, self.processor, self.context_len = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            torch_dtype=self.torch_dtype,
            device=self.device
        )
        print("Model loaded successfully!")
        if hasattr(self.model, 'device'):
            target_device = self.model.device
        else:
            target_device = next(self.model.parameters()).device
        self.model = self.model.to(target_device)
        self.device = str(target_device)
        # print("@@@@@@@@@@@Model", next(self.model.parameters()).device)


    def _prepare_prompt(self, query: str, clicks: Optional[List[float]] = None) -> Tuple[str, torch.Tensor]:
        """Prepare prompt for LLaVA."""
        qs = query

        # Handle coordinate clicks if provided
        if clicks:
            coord_list = [round(c, 3) for c in clicks[:3]]
            qs = qs.replace("[coord]", "<boxes>")
            clicks_tensor = torch.tensor([coord_list])
        else:
            clicks_tensor = torch.zeros((0, 3))

        # Add image tokens
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
            else:
                qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        # Build conversation
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        return prompt, clicks_tensor

    def recognize_image(
        self,
        image_path: str,
        query: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        num_beams: int = 1,
    ) -> SceneRecognitionResult:
        """
        Perform scene recognition on an image.

        Args:
            image_path: Path to image file
            query: Query prompt (uses default SCENE_UNDERSTANDING_PROMPT if None)
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams for beam search

        Returns:
            SceneRecognitionResult with raw output and parsed predicates
        """
        if query is None:
            query = SCENE_UNDERSTANDING_PROMPT

        # Load and process image
        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        images_tensor = process_images(
            [image],
            self.processor['image'],
            self.model.config
        ).to(self.model.device, dtype=self.torch_dtype)

        # Prepare prompt
        prompt, clicks_tensor = self._prepare_prompt(query)

        input_ids = (
            tokenizer_special_token(prompt, self.tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        # print("@@@@@@@@@@", input_ids.device)
        # print("@@@@@@@@@@@@@@", next(self.model.parameters()).device)
        # print( "@@@@@@@@@@@@@@@@@", images_tensor.device)
        # for name , param in self.model.named_parameters():
        #     print("##", name, param.device)
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                depths=None,
                poses=None,
                intrinsics=None,
                clicks=None,
                image_sizes=[image_size],
                do_sample=temperature > 0,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Parse to predicates
        predicates = self._extract_predicates(output)

        return SceneRecognitionResult(
            raw_output=output,
            predicates=predicates,
            entities=self._parse_entities(predicates),
            image_path=image_path,
            query=query
        )

    def recognize_video(
        self,
        video_path: str,
        query: Optional[str] = None,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
        num_beams: int = 1,
    ) -> SceneRecognitionResult:
        """
        Perform scene recognition on a video.

        Args:
            video_path: Path to video file
            query: Query prompt
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate
            num_beams: Number of beams for beam search

        Returns:
            SceneRecognitionResult
        """
        if query is None:
            query = SCENE_UNDERSTANDING_PROMPT

        # Process video
        videos_dict = process_videos(
            video_path,
            self.processor['video'],
            mode='random',
            device=self.model.device,
            text=query
        )

        images_tensor = videos_dict['images'].to(self.model.device, dtype=self.torch_dtype)
        depths_tensor = videos_dict['depths'].to(self.model.device, dtype=self.torch_dtype)
        poses_tensor = videos_dict['poses'].to(self.model.device, dtype=self.torch_dtype)
        intrinsics_tensor = videos_dict['intrinsics'].to(self.model.device, dtype=self.torch_dtype)

        # Prepare prompt
        prompt, clicks_tensor = self._prepare_prompt(query)

        input_ids = (
            tokenizer_special_token(prompt, self.tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                depths=depths_tensor,
                poses=poses_tensor,
                intrinsics=intrinsics_tensor,
                clicks=clicks_tensor.to(self.model.device, dtype=self.torch_dtype),
                image_sizes=None,
                do_sample=temperature > 0,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Parse to predicates
        predicates = self._extract_predicates(output)

        return SceneRecognitionResult(
            raw_output=output,
            predicates=predicates,
            entities=self._parse_entities(predicates),
            image_path=video_path,
            query=query
        )

    def _extract_predicates(self, raw_output: str) -> str:
        """
        Extract Prolog-style predicates from LLaVA output.

        Looks for patterns like:
            object(id, category, [types]).
            position(id, x, y, z).
            attribute(id, name, value).
        """
        import re

        # Find predicate-like lines
        predicate_pattern = re.compile(
            r'(object|position|attribute|relation|near|on|above|touching)'
            r'\([^)]+\)\s*\.',
            re.IGNORECASE
        )

        matches = predicate_pattern.findall(raw_output)

        if matches:
            # Extract all matching lines
            lines = []
            for match in re.finditer(predicate_pattern, raw_output):
                lines.append(match.group(0))
            return '\n'.join(lines)

        # If no structured predicates found, try to parse from description
        return self._heuristic_parse(raw_output)

    def _heuristic_parse(self, raw_output: str) -> str:
        """
        Heuristically parse free-form text into Prolog predicates.
        Fallback when LLaVA doesn't output structured format.
        """
        import re

        predicates = []
        lines = raw_output.lower().split('\n')

        obj_id = 1
        for line in lines:
            # Look for object mentions
            if any(word in line for word in ['gasoline', 'gas can', 'fuel']):
                predicates.append(f'object({obj_id}, gasoline_can, flammable).')
                obj_id += 1
            elif any(word in line for word in ['stove', 'oven', 'fire']):
                predicates.append(f'object({obj_id}, stove, fire_source).')
                obj_id += 1
            elif any(word in line for word in ['knife', 'blade', 'sharp']):
                predicates.append(f'object({obj_id}, knife, sharp).')
                obj_id += 1
            elif any(word in line for word in ['water', 'liquid']):
                predicates.append(f'object({obj_id}, water, liquid).')
                obj_id += 1
            elif any(word in line for word in ['person', 'human', 'child']):
                predicates.append(f'object({obj_id}, person, is_person).')
                obj_id += 1

        return '\n'.join(predicates) if predicates else raw_output

    def _parse_entities(self, predicates: str) -> List[Dict]:
        """Parse predicates into structured entity list."""
        entities = []
        import re

        # Parse object predicates
        obj_pattern = re.compile(
            r'object\((\w+),\s*(\w+),\s*\[?([^\]]*)\]?\)',
            re.IGNORECASE
        )

        for match in obj_pattern.finditer(predicates):
            obj_id = match.group(1)
            category = match.group(2)
            types_str = match.group(3)
            types = [t.strip() for t in types_str.split(',') if t.strip()]

            entities.append({
                'id': obj_id,
                'category': category,
                'semantic_types': types
            })

        return entities

    def parse_to_predicates(
        self,
        result: SceneRecognitionResult,
        builder=None
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Convert SceneRecognitionResult to grounded facts using SceneGraphBuilder.

        Args:
            result: Output from recognize_image/recognize_video
            builder: Optional SceneGraphBuilder instance

        Returns:
            grounded_facts: { "near(1,2)": tensor(0.92), ... }
            entity_ids: ["1", "2", ...]
        """
        if builder is None:
            from scene_graph_builder import SceneGraphBuilder
            builder = SceneGraphBuilder()

        return builder.build(result.predicates)


class MockLLaVA3DRecognizer:
    """
    Mock recognizer for testing without actual LLaVA-3D model.
    Returns predefined predicates for development and testing.
    """

    DEFAULT_PREDICATES = """
object(1, gasoline_can, flammable).
object(2, stove, fire_source).
object(3, cloth, [flammable, is_cloth]).
object(4, knife, sharp).
object(5, water, liquid).
object(6, power_strip, electrical).

position(1, 1.2, 0.5, 0.8).
position(2, 1.0, 0.5, 0.8).
position(3, 1.15, 0.55, 0.78).
position(4, 2.0, 0.4, 0.9).
position(5, 0.8, 0.5, 1.0).
position(6, 0.85, 0.3, 1.15).

attribute(2, temperature, 200).
attribute(4, sharpness, 0.95).
"""

    def __init__(self, *args, **kwargs):
        pass

    def recognize_image(self, image_path: str, **kwargs) -> SceneRecognitionResult:
        return SceneRecognitionResult(
            raw_output="Mock output",
            predicates=self.DEFAULT_PREDICATES,
            entities=[
                {'id': '1', 'category': 'gasoline_can', 'semantic_types': ['flammable']},
                {'id': '2', 'category': 'stove', 'semantic_types': ['fire_source']},
            ],
            image_path=image_path,
            query="Mock query"
        )

    def recognize_video(self, video_path: str, **kwargs) -> SceneRecognitionResult:
        return self.recognize_image(video_path)

    def parse_to_predicates(self, result: SceneRecognitionResult, builder=None):
        if builder is None:
            from scene_graph_builder import SceneGraphBuilder
            builder = SceneGraphBuilder()
        return builder.build(result.predicates)
