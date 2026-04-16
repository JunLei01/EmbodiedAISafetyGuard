"""
Convert ProcTHOR Kitchen Scene to Predicates
==============================================
Loads ProcTHOR-10k dataset using prior and converts kitchen scenes to predicates format.

Usage:
    python convert_kitchen_scene.py
"""

import sys
sys.path.insert(0, '/data/junlei/NPG')
sys.path.insert(0, '/data/junlei/NPG/test_src')

import torch
from collections import defaultdict

# Import the pipeline components
from procthor_adapter import ProcTHORAdapter, ProcTHORObject, ProcTHORScene
from utils.scene_graph_builder import SceneGraphBuilder
from utils.differentiable_safety_reasoner import DifferentiableSafetyReasoner
from utils.safety_knowledge_base import SafetyCategory, Severity


def load_procthor_dataset():
    """Load ProcTHOR-10k dataset using prior."""
    print("Loading ProcTHOR-10k dataset...")
    try:
        import prior
        dataset = prior.load_dataset("procthor-10k")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure prior is installed: pip install prior")
        raise


def find_kitchen_scenes(dataset_split, max_check=100):
    """Find kitchen scenes in the dataset."""
    kitchen_keywords = {
        'Microwave', 'StoveBurner', 'CoffeeMaker', 'Fridge', 'Refrigerator',
        'Toaster', 'Kettle', 'Sink', 'Knife', 'Pot', 'Pan', 'Oven',
        'DishSponge', 'SoapBottle', 'CounterTop', ' Cabinet'
    }

    kitchen_scenes = []

    for i in range(min(max_check, len(dataset_split))):
        scene_data = dataset_split[i]
        objects = scene_data.get('objects', [])
        obj_types = set(obj.get('objectType', '') for obj in objects)

        kitchen_count = len(obj_types & kitchen_keywords)

        if kitchen_count >= 2 or ('Microwave' in obj_types and 'Sink' in obj_types):
            kitchen_scenes.append((i, scene_data, obj_types))
            if len(kitchen_scenes) >= 5:
                break

    return kitchen_scenes


def create_semantic_mapping():
    """Create comprehensive semantic type mapping for ProcTHOR objects."""
    return {
        # Fire/Heat hazards
        'Microwave': ['fire_source', 'electrical'],
        'CoffeeMaker': ['hot', 'electrical'],
        'Toaster': ['fire_source', 'hot', 'electrical'],
        'StoveBurner': ['fire_source', 'hot', 'electrical'],
        'Kettle': ['hot', 'electrical'],
        'Pot': ['hot', 'container'],
        'Pan': ['hot', 'container'],
        'Oven': ['fire_source', 'hot', 'electrical'],
        'Candle': ['fire_source', 'flammable'],
        'Lighter': ['fire_source', 'flammable'],

        # Electrical appliances
        'Fridge': ['electrical', 'heavy'],
        'Refrigerator': ['electrical', 'heavy'],
        'Dishwasher': ['electrical', 'is_wet'],
        'WashingMachine': ['electrical', 'is_wet'],
        'Dryer': ['electrical', 'hot'],
        'VacuumCleaner': ['electrical'],
        'Blender': ['electrical'],

        # Sharp objects
        'Knife': ['sharp', 'weapon'],
        'ButterKnife': ['sharp'],
        'SteakKnife': ['sharp', 'weapon'],
        'Scissors': ['sharp'],
        'Fork': ['small', 'sharp'],
        'WineBottle': ['fragile', 'glass'],
        'Glass': ['fragile', 'glass'],
        'Cup': ['fragile'],
        'Plate': ['fragile', 'breakable'],
        'Bowl': ['fragile'],
        'Mug': ['fragile', 'is_container'],
        'Vase': ['fragile'],
        'Mirror': ['fragile', 'glass'],

        # Chemical/Cleaning
        'SprayBottle': ['chemical', 'toxic', 'liquid'],
        'SoapBottle': ['chemical', 'liquid'],
        'CleaningSolution': ['chemical', 'toxic', 'liquid'],
        'Bleach': ['chemical', 'toxic', 'liquid'],

        # Containers
        'Box': ['is_container'],
        'Jar': ['is_container'],
        'Can': ['is_container'],
        'Bottle': ['is_container', 'fragile', 'glass'],
        'Bucket': ['is_container'],
        'Pot': ['is_container', 'hot'],
        'Pan': ['is_container', 'hot'],

        # Food items
        'Apple': ['small'],
        'Banana': ['small'],
        'Bread': ['small'],
        'Lettuce': ['small'],
        'Tomato': ['small'],
        'Potato': ['small'],
        'Egg': ['small', 'fragile'],
        'Orange': ['small'],
        'Grape': ['small'],
        'Sandwich': ['small'],
        'SandwichWrapped': ['small'],

        # Kitchen utensils
        'Spoon': ['small'],
        'Spatula': ['small'],
        'SaltShaker': ['small'],
        'PepperShaker': ['small'],
        'Ladle': ['small'],

        # Misc small items
        'CellPhone': ['small', 'electrical'],
        'Watch': ['small'],
        'Pen': ['small'],
        'Pencil': ['small'],
        'Book': ['small'],
        'Newspaper': ['small'],
        'Remote': ['small'],
        'Key': ['small'],

        # Furniture
        'Chair': ['furniture'],
        'BarStool': ['furniture'],
        'DiningTable': ['furniture', 'heavy'],
        'CoffeeTable': ['furniture'],
        'SideTable': ['furniture'],
        'CounterTop': ['furniture'],
        'ShelvingUnit': ['furniture'],
        'Shelf': ['furniture'],

        # Fixtures
        'Sink': ['heavy', 'is_wet'],
        'SinkBasin': ['is_wet'],
        'Bathtub': ['heavy', 'is_wet'],
        'Toilet': ['heavy'],
        'ShowerHead': ['is_wet'],
        'Faucet': ['is_wet'],

        # Decor
        'PictureFrame': ['fragile'],
        'Painting': ['fragile'],
        'HousePlant': [],
        'Plant': [],

        # Person
        'Person': ['is_person'],
    }


def convert_to_predicates(scene, semantic_mapping):
    """Convert a ProcTHOR scene to predicate format."""

    def get_types(category):
        return semantic_mapping.get(category, [])

    lines = [
        f"% ProcTHOR scene: {scene.scene_id}",
        f"% Room type: {scene.room_type}",
        f"% Objects: {scene.num_objects}",
        "",
        "% ── Objects in the kitchen ──",
    ]

    for obj in scene.objects:
        types = get_types(obj.category)
        if len(types) == 0:
            types_str = "[]"
        elif len(types) == 1:
            types_str = types[0]
        else:
            types_str = "[" + ", ".join(types) + "]"

        lines.append(f"object({obj.obj_id}, {obj.category}, {types_str}).")

    # Attributes
    lines.append("")
    lines.append("% ── Attributes ──")

    for obj in scene.objects:
        types = get_types(obj.category)
        if 'hot' in types or 'fire_source' in types:
            lines.append(f"attribute({obj.obj_id}, temperature, 100).")
        if 'electrical' in types:
            lines.append(f"attribute({obj.obj_id}, powered, 1).")
        if 'is_wet' in types:
            lines.append(f"attribute({obj.obj_id}, wet, 1).")

    # Positions
    lines.append("")
    lines.append("% ── 3D Positions (meters) ──")

    for obj in scene.objects:
        x, y, z = obj.position_3d
        lines.append(f"position({obj.obj_id}, {x:.3f}, {y:.3f}, {z:.3f}).")

    return "\n".join(lines)


def analyze_scene(scene, semantic_mapping):
    """Analyze scene objects and their semantic types."""
    print(f"\nScene Analysis:")
    print(f"  ID: {scene.scene_id}")
    print(f"  Room type: {scene.room_type}")
    print(f"  Total objects: {scene.num_objects}")
    print(f"\nObject breakdown:")

    hazard_objects = []
    for obj in scene.objects:
        types = semantic_mapping.get(obj.category, [])
        hazard_types = [t for t in types if t in ['sharp', 'hot', 'fire_source', 'chemical', 'toxic', 'electrical', 'flammable']]
        if hazard_types:
            hazard_objects.append((obj, hazard_types))

    print(f"  Hazard objects: {len(hazard_objects)}")
    for obj, hazards in hazard_objects:
        print(f"    - {obj.category} ({obj.obj_id}): {hazards}")

    return hazard_objects


def run_safety_pipeline(scene, semantic_mapping):
    """Run the complete neuro-symbolic safety pipeline."""
    print("\n" + "=" * 70)
    print("Running Safety Pipeline")
    print("=" * 70)

    # Stage 1: Build scene graph
    print("\n[Stage 1] Building scene graph...")
    builder = SceneGraphBuilder(require_grad_positions=False)

    # Create parsed scene with custom semantic types
    from utils.scene_graph_builder import ParsedScene, SceneObject
    parsed = ParsedScene()

    for obj in scene.objects:
        semantic_types = set(semantic_mapping.get(obj.category, []))
        parsed.objects[obj.obj_id] = SceneObject(
            obj_id=obj.obj_id,
            category=obj.category,
            semantic_types=semantic_types,
            position_3d=obj.position_3d,
            attributes={},
            detection_confidence=1.0
        )

    grounded_facts, entity_ids = builder.build_from_parsed(parsed)

    print(f"  Entities: {len(entity_ids)}")
    print(f"  Grounded facts: {len(grounded_facts)}")

    # Count spatial relations
    spatial_count = sum(1 for k in grounded_facts.keys()
                       if any(r in k for r in ['near(', 'touching(', 'above(', 'below(', 'left_of(', 'right_of(']))
    print(f"  Spatial relations: {spatial_count}")

    # Stage 2: Safety reasoning
    print("\n[Stage 2] Running differentiable safety reasoning...")
    reasoner = DifferentiableSafetyReasoner(activation_threshold=0.3)
    result = reasoner.reason(grounded_facts, entity_ids)

    print(f"  Activated rules: {result.num_activated}")
    print(f"  Critical: {sum(1 for r in result.activated_rules if r.severity == Severity.CRITICAL)}")
    print(f"  High: {sum(1 for r in result.activated_rules if r.severity == Severity.HIGH)}")
    print(f"  Medium: {sum(1 for r in result.activated_rules if r.severity == Severity.MEDIUM)}")
    print(f"  Low: {sum(1 for r in result.activated_rules if r.severity == Severity.LOW)}")

    if result.activated_rules:
        print("\n  Activated rules:")
        for rule in result.activated_rules[:10]:
            print(f"    [{rule.severity.name:8}] {rule.rule_name}: P={rule.prob_value:.3f}")
            print(f"             {rule.grounded_description or rule.natural_language}")

    # Stage 3: Action checking examples
    print("\n[Stage 3] Example action safety checks:")

    test_actions = [
        ('move_towards', 'Knife', 'Person'),
        ('place_near', 'WineBottle', 'Edge'),
        ('turn_on', 'Microwave', None),
        ('pour', 'SprayBottle', None),
    ]

    for action, actor_cat, target_cat in test_actions:
        # Find entities
        actor_id = None
        target_id = None

        for obj in scene.objects:
            if obj.category == actor_cat:
                actor_id = obj.obj_id
            if target_cat and obj.category == target_cat:
                target_id = obj.obj_id

        if actor_id:
            violations = result.check_action(action, actor_id, target_id)
            status = "BLOCKED" if violations else "ALLOWED"
            print(f"\n  [{status}] {action}({actor_cat}, {target_cat or ''})")
            for v in violations:
                print(f"    - Violates: {v.rule_name} ({v.severity.name})")

    return result


def main():
    # Load dataset
    dataset = load_procthor_dataset()
    train_split = dataset['train']

    # Find kitchen scenes
    print("\nSearching for kitchen scenes...")
    kitchen_scenes = find_kitchen_scenes(train_split, max_check=200)

    if not kitchen_scenes:
        print("No kitchen scenes found! Using scene 0.")
        kitchen_scenes = [(0, train_split[0], set())]

    print(f"\nFound {len(kitchen_scenes)} kitchen scenes")

    # Use the first kitchen scene
    scene_idx, scene_data, obj_types = kitchen_scenes[0]
    print(f"\nUsing scene at index {scene_idx}")
    print(f"Object types: {sorted(obj_types)[:15]}...")

    # Create adapter and load scene
    adapter = ProcTHORAdapter(train_split)

    # Get scene ID
    scene_id = adapter.list_scenes()[scene_idx]
    scene = adapter.load_scene(scene_id)

    # Create semantic mapping
    semantic_mapping = create_semantic_mapping()

    # Analyze scene
    analyze_scene(scene, semantic_mapping)

    # Convert to predicates
    print("\n" + "=" * 70)
    print("GENERATED PREDICATES:")
    print("=" * 70)

    predicates = convert_to_predicates(scene, semantic_mapping)
    print(predicates)

    # Save to file
    output_file = "/data/junlei/NPG/kitchen_scene_predicates.pl"
    with open(output_file, 'w') as f:
        f.write(predicates)
    print(f"\n\nSaved to: {output_file}")

    # Run safety pipeline
    result = run_safety_pipeline(scene, semantic_mapping)


if __name__ == "__main__":
    main()
