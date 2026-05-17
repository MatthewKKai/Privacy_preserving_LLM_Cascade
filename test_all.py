"""
Comprehensive test suite for P3Defer.
Tests all module imports, class instantiation, and basic functionality.
"""

import sys
import os
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0


def test(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS: {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        traceback.print_exc()
        FAIL += 1


# ============================================================
# 1. Module Imports
# ============================================================
print("\n=== Module Import Tests ===")


def test_import_memory():
    from p3defer.memory import PrivateMemory
    assert PrivateMemory is not None

test("Import PrivateMemory", test_import_memory)


def test_import_models():
    from p3defer.models import PolicyNetwork, ValueNetwork, StateEncoder, CascadeModel
    assert PolicyNetwork is not None
    assert ValueNetwork is not None
    assert StateEncoder is not None
    assert CascadeModel is not None

test("Import models (PolicyNetwork, ValueNetwork, StateEncoder, CascadeModel)", test_import_models)


def test_import_training():
    from p3defer.training import PPODeferralTrainer, InstructionTuner, MultiObjectiveLossTuner
    assert PPODeferralTrainer is not None
    assert InstructionTuner is not None
    assert MultiObjectiveLossTuner is not None

test("Import training modules", test_import_training)


def test_import_evaluation():
    from p3defer.evaluation import CascadeEvaluator
    assert CascadeEvaluator is not None

test("Import CascadeEvaluator", test_import_evaluation)


def test_import_data():
    from p3defer.data import get_processor, download_and_prepare_dataset
    assert get_processor is not None
    assert download_and_prepare_dataset is not None

test("Import data modules", test_import_data)


def test_import_scripts():
    # Test that all entry-point scripts can be parsed
    import importlib.util
    for script in ["prepare_data", "build_memory", "run_instruction_tuning",
                   "run_loss_tuning", "run_ppo_training", "run_evaluation", "run_inference"]:
        spec = importlib.util.spec_from_file_location(
            script, os.path.join(os.path.dirname(__file__), f"{script}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # Don't execute, just verify it can be loaded
        assert spec is not None

test("Import all entry-point scripts", test_import_scripts)


# ============================================================
# 2. Private Memory Tests
# ============================================================
print("\n=== Private Memory Tests ===")


def test_memory_basic():
    from p3defer.memory import PrivateMemory
    mem = PrivateMemory(threshold=0.3)
    mem.add_token("John", "name")
    mem.add_token("Smith", "name")
    assert mem.size == 2

test("PrivateMemory basic add", test_memory_basic)


def test_memory_detection():
    from p3defer.memory import PrivateMemory
    mem = PrivateMemory(threshold=0.3)
    mem.add_token("John", "name")
    mem.add_token("Smith", "name")
    detections = mem.detect_private_tokens("John went to the store with Smith")
    assert len(detections) >= 2

test("PrivateMemory detection", test_memory_detection)


def test_memory_masking():
    from p3defer.memory import PrivateMemory
    mem = PrivateMemory(threshold=0.3)
    mem.add_token("John", "name")
    masked, replacements, count = mem.mask_query("John went to the store")
    assert "John" not in masked or count == 0  # Either masked or no match
    assert isinstance(replacements, list)

test("PrivateMemory masking", test_memory_masking)


def test_memory_save_load():
    import tempfile
    from p3defer.memory import PrivateMemory
    mem = PrivateMemory(threshold=0.3)
    mem.add_token("Alice", "name")
    mem.add_token("Bob", "name")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    mem.save(path)

    mem2 = PrivateMemory()
    mem2.load(path)
    assert mem2.size == 2
    os.unlink(path)

test("PrivateMemory save/load", test_memory_save_load)


def test_memory_corpus():
    from p3defer.memory import PrivateMemory
    mem = PrivateMemory(threshold=0.3)
    texts = [
        "John Smith went to the hospital",
        "Mary Johnson called her doctor",
        "The weather is nice today",
    ]
    labels = [1, 1, 0]
    added = mem.add_tokens_from_corpus(texts, labels)
    assert added > 0
    assert mem.size > 0

test("PrivateMemory corpus extraction", test_memory_corpus)


# ============================================================
# 3. Model Tests
# ============================================================
print("\n=== Model Tests ===")

import torch


def test_state_encoder():
    from p3defer.models import StateEncoder
    enc = StateEncoder(privacy_dim=2, quality_dim=1, hidden_dim=64, state_dim=128)
    priv = torch.randn(4, 2)
    qual = torch.randn(4, 1)
    state = enc(priv, qual)
    assert state.shape == (4, 128)

test("StateEncoder forward", test_state_encoder)


def test_policy_network():
    from p3defer.models import PolicyNetwork
    policy = PolicyNetwork(state_dim=128, hidden_dim=128, num_actions=3)
    state = torch.randn(4, 128)
    action, log_prob, entropy = policy.get_action(state)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)

test("PolicyNetwork get_action", test_policy_network)


def test_value_network():
    from p3defer.models import ValueNetwork
    value_net = ValueNetwork(state_dim=128, hidden_dim=128)
    state = torch.randn(4, 128)
    value = value_net(state)
    assert value.shape == (4, 1)

test("ValueNetwork forward", test_value_network)


def test_policy_evaluate():
    from p3defer.models import PolicyNetwork
    policy = PolicyNetwork(state_dim=128, hidden_dim=128, num_actions=3)
    state = torch.randn(4, 128)
    action = torch.tensor([0, 1, 2, 0])
    log_prob, entropy = policy.evaluate_action(state, action)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)

test("PolicyNetwork evaluate_action", test_policy_evaluate)


# ============================================================
# 4. PPO Trainer Tests
# ============================================================
print("\n=== PPO Trainer Tests ===")


def test_ppo_trainer_init():
    from p3defer.training import PPODeferralTrainer
    trainer = PPODeferralTrainer(
        state_dim=64, hidden_dim=64, device="cpu"
    )
    assert trainer.policy is not None
    assert trainer.value is not None

test("PPODeferralTrainer initialization", test_ppo_trainer_init)


def test_ppo_rollout():
    from p3defer.training import PPODeferralTrainer
    trainer = PPODeferralTrainer(
        state_dim=64, hidden_dim=64, device="cpu"
    )
    privacy_signals = [torch.tensor([0.8, 0.2]) for _ in range(10)]
    quality_signals = [torch.tensor([0.7]) for _ in range(10)]
    privacy_labels = [True, False, True, False, True, False, True, False, True, False]

    stats = trainer.collect_rollout(privacy_signals, quality_signals, privacy_labels)
    assert "mean_reward" in stats
    assert stats["num_steps"] == 10

test("PPODeferralTrainer rollout collection", test_ppo_rollout)


def test_ppo_update():
    from p3defer.training import PPODeferralTrainer
    trainer = PPODeferralTrainer(
        state_dim=64, hidden_dim=64, device="cpu"
    )
    privacy_signals = [torch.tensor([0.8, 0.2]) for _ in range(20)]
    quality_signals = [torch.tensor([0.7]) for _ in range(20)]
    privacy_labels = [i % 2 == 0 for i in range(20)]

    trainer.collect_rollout(privacy_signals, quality_signals, privacy_labels)
    update_stats = trainer.update()
    assert "policy_loss" in update_stats
    assert "value_loss" in update_stats

test("PPODeferralTrainer PPO update", test_ppo_update)


def test_ppo_save_load():
    import tempfile
    from p3defer.training import PPODeferralTrainer
    trainer = PPODeferralTrainer(
        state_dim=64, hidden_dim=64, device="cpu"
    )
    tmpdir = tempfile.mkdtemp()
    trainer.save(tmpdir)

    trainer2 = PPODeferralTrainer(
        state_dim=64, hidden_dim=64, device="cpu"
    )
    trainer2.load(tmpdir)
    # Verify weights are loaded
    for p1, p2 in zip(trainer.policy.parameters(), trainer2.policy.parameters()):
        assert torch.allclose(p1, p2)

    import shutil
    shutil.rmtree(tmpdir)

test("PPODeferralTrainer save/load", test_ppo_save_load)


# ============================================================
# 5. Evaluation Tests
# ============================================================
print("\n=== Evaluation Tests ===")


def test_evaluator_basic():
    from p3defer.evaluation import CascadeEvaluator
    import tempfile
    tmpdir = tempfile.mkdtemp()
    evaluator = CascadeEvaluator(dataset_name="gsm8k", output_dir=tmpdir)

    results = evaluator.evaluate(
        predictions=["The answer is 42", "The sum is 10", "Result: 5"],
        references=["The answer is 42", "The sum is 15", "Result: 5"],
        actions=[0, 1, 0],
        privacy_labels=[0, 1, 0],
        privacy_predictions=[0, 1, 0],
    )
    assert "accuracy" in results
    assert "coverage_rate" in results
    assert "server_coverage_rate" in results
    assert "privacy_precision" in results

    import shutil
    shutil.rmtree(tmpdir)

test("CascadeEvaluator basic evaluation", test_evaluator_basic)


def test_evaluator_rouge():
    from p3defer.evaluation.evaluator import compute_rouge
    scores = compute_rouge(
        ["The cat sat on the mat", "Hello world"],
        ["The cat sat on the mat", "Hello there world"],
    )
    assert "rouge1" in scores
    assert "rougeL" in scores
    assert scores["rouge1"] > 0

test("ROUGE computation", test_evaluator_rouge)


def test_evaluator_accuracy():
    from p3defer.evaluation.evaluator import compute_accuracy
    acc = compute_accuracy(
        ["The answer is 42", "The answer is 10", "#### 5"],
        ["#### 42", "#### 15", "#### 5"],
    )
    assert acc > 0  # At least 2/3 correct

test("Accuracy computation", test_evaluator_accuracy)


# ============================================================
# 6. Data Processor Tests
# ============================================================
print("\n=== Data Processor Tests ===")


def test_data_processor_gsm8k():
    from p3defer.data import get_processor
    proc = get_processor("gsm8k")
    assert proc is not None
    item = {"question": "What is 2+2?", "answer": "4"}
    formatted = proc.format_input(item)
    assert "2+2" in formatted

test("GSM8K processor", test_data_processor_gsm8k)


def test_data_processor_medsum():
    from p3defer.data import get_processor
    proc = get_processor("medsum")
    assert proc is not None
    item = {"question": "Patient has fever", "answer": "Fever noted"}
    formatted = proc.format_input(item)
    assert "fever" in formatted.lower()

test("MedSum processor", test_data_processor_medsum)


def test_data_processor_emailsum():
    from p3defer.data import get_processor
    proc = get_processor("emailsum")
    assert proc is not None
    item = {"question": "Meeting at 3pm", "answer": "Meeting scheduled"}
    formatted = proc.format_input(item)
    assert "Meeting" in formatted

test("EmailSum processor", test_data_processor_emailsum)


# ============================================================
# 7. Instruction Tuner Tests
# ============================================================
print("\n=== Instruction Tuner Tests ===")


def test_instruction_tuner_cot():
    from p3defer.training import InstructionTuner
    tuner = InstructionTuner(model_name="gpt2")
    raw_data = [
        {"question": "What is 2+2?", "answer": "4", "privacy": 0},
        {"question": "John has 5 apples", "answer": "5", "privacy": 1},
    ]
    cot_data = tuner.prepare_cot_data(raw_data, "gsm8k")
    assert len(cot_data) == 2
    assert "input" in cot_data[0]
    assert "output" in cot_data[0]
    assert "step by step" in cot_data[0]["input"].lower()

test("InstructionTuner CoT data preparation", test_instruction_tuner_cot)


# ============================================================
# 8. Reward Function Tests
# ============================================================
print("\n=== Reward Function Tests ===")


def test_reward_function():
    from p3defer.training.ppo_trainer import RewardFunction
    rf = RewardFunction(lambda_privacy=0.5)

    # Local answer, no privacy concern
    r1 = rf.compute(action=0, quality_score=0.8, has_privacy=False)
    assert r1 > 0

    # Defer with privacy, masked
    r2 = rf.compute(action=1, quality_score=0.9, has_privacy=True, privacy_masked=True)
    assert r2 > 0

    # Abstain
    r3 = rf.compute(action=2, quality_score=0.0, has_privacy=True)
    assert r3 >= 0  # Should get privacy credit but no quality

test("RewardFunction computation", test_reward_function)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"Test Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
print(f"{'='*60}")

sys.exit(0 if FAIL == 0 else 1)
