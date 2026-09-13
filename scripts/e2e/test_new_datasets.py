"""Test that all new datasets conform to spec.

For each new dataset:
1. Import from registry
2. Instantiate
3. Check len() or iteration works
4. Retrieve 2-3 samples
5. Validate sample types
"""

import sys
from theseus.data.datasets.dataset import ChatTurn


def check_chat_template(sample: object, name: str) -> bool:
    """Validate a ChatTemplate sample."""
    if not isinstance(sample, list):
        print(f"  FAIL: {name} returned {type(sample)}, expected list")
        return False
    for i, turn in enumerate(sample):
        if not isinstance(turn, ChatTurn):
            print(f"  FAIL: {name}[{i}] is {type(turn)}, expected ChatTurn")
            return False
        if not isinstance(turn.role, str) or not isinstance(turn.message, str):
            print(f"  FAIL: {name}[{i}] has non-string role/message")
            return False
    return True


def test_cfq() -> bool:
    print("\n=== Testing CFQ ===")
    from theseus.data.datasets.cfq import CFQ

    ds = CFQ(split="train", config="mcd1")
    print(f"  len: {len(ds)}")
    assert len(ds) > 0, "CFQ has no items"

    for i in range(min(2, len(ds))):
        sample = ds[i]
        if not check_chat_template(sample, f"CFQ[{i}]"):
            return False
        print(f"  sample[{i}] roles: {[t.role for t in sample]}")
        print(f"  user msg preview: {sample[0].message[:100]}...")
        print(f"  assistant msg preview: {sample[1].message[:100]}...")

    print("  PASS")
    return True


def test_clutrr() -> bool:
    print("\n=== Testing CLUTRR ===")
    from theseus.data.datasets.clutrr import CLUTRR

    ds = CLUTRR(split="train", config="gen_train234_test2to10")
    print(f"  len: {len(ds)}")
    assert len(ds) > 0, "CLUTRR has no items"

    for i in range(min(2, len(ds))):
        sample = ds[i]
        if not check_chat_template(sample, f"CLUTRR[{i}]"):
            return False
        print(f"  sample[{i}] roles: {[t.role for t in sample]}")
        print(f"  user msg preview: {sample[0].message[:100]}...")
        print(f"  assistant answer: {sample[1].message}")

    print("  PASS")
    return True


def test_pg19() -> bool:
    print("\n=== Testing PG19 (streaming) ===")
    from theseus.data.datasets.pg19 import PG19

    ds = PG19(split="train")
    count = 0
    for item in ds:
        assert isinstance(item, str), f"PG19 yielded {type(item)}, expected str"
        print(f"  sample[{count}] length: {len(item)} chars")
        print(f"  preview: {item[:100]}...")
        count += 1
        if count >= 2:
            break

    assert count > 0, "PG19 yielded no items"
    print("  PASS")
    return True


def test_ccaligned() -> bool:
    print("\n=== Testing CCAligned (streaming) ===")
    from theseus.data.datasets.ccaligned import CCAligned

    ds = CCAligned(config="fr_XX")
    count = 0
    for item in ds:
        assert isinstance(item, str), f"CCAligned yielded {type(item)}, expected str"
        print(f"  sample[{count}] length: {len(item)} chars")
        print(f"  preview: {item[:100]}...")
        count += 1
        if count >= 3:
            break

    assert count > 0, "CCAligned yielded no items"
    print("  PASS")
    return True


def test_pes2o() -> bool:
    print("\n=== Testing Pes2O (streaming) ===")
    from theseus.data.datasets.pes2o import Pes2O

    ds = Pes2O()
    count = 0
    for item in ds:
        assert isinstance(item, str), f"Pes2O yielded {type(item)}, expected str"
        print(f"  sample[{count}] length: {len(item)} chars")
        print(f"  preview: {item[:100]}...")
        count += 1
        if count >= 3:
            break

    assert count > 0, "Pes2O yielded no items"
    print("  PASS")
    return True


def test_pile() -> bool:
    print("\n=== Testing Pile (streaming) ===")
    from theseus.data.datasets.pile import Pile

    ds = Pile()
    count = 0
    for item in ds:
        assert isinstance(item, str), f"Pile yielded {type(item)}, expected str"
        print(f"  sample[{count}] length: {len(item)} chars")
        print(f"  preview: {item[:100]}...")
        count += 1
        if count >= 3:
            break

    assert count > 0, "Pile yielded no items"
    print("  PASS")
    return True


def test_longhealth() -> bool:
    print("\n=== Testing LongHealth ===")
    from theseus.data.datasets.longhealth import LongHealth

    ds = LongHealth()
    print(f"  len: {len(ds)}")
    assert len(ds) > 0, "LongHealth has no items"

    for i in range(min(2, len(ds))):
        sample = ds[i]
        if not check_chat_template(sample, f"LongHealth[{i}]"):
            return False
        print(f"  sample[{i}] roles: {[t.role for t in sample]}")
        print(f"  user msg length: {len(sample[0].message)} chars")
        print(f"  assistant answer: {sample[1].message}")

    print("  PASS")
    return True


def test_mtob() -> bool:
    print("\n=== Testing MTOB ===")
    from theseus.data.datasets.mtob import MTOB

    ds = MTOB(split="train", config="en-kgv")
    print(f"  len: {len(ds)}")
    assert len(ds) > 0, "MTOB has no items"

    for i in range(min(2, len(ds))):
        sample = ds[i]
        if not check_chat_template(sample, f"MTOB[{i}]"):
            return False
        print(f"  sample[{i}] roles: {[t.role for t in sample]}")
        print(f"  user msg preview: {sample[0].message[:100]}...")
        print(f"  assistant answer: {sample[1].message}")

    print("  PASS")
    return True


def test_registry() -> bool:
    """Test that all new datasets are in the registry."""
    print("\n=== Testing Registry ===")
    from theseus.registry import DATASETS

    expected = [
        "ccaligned",
        "cfq",
        "clutrr",
        "longhealth",
        "mtob",
        "pes2o",
        "pg19",
        "pile",
    ]
    for name in expected:
        assert name in DATASETS, f"  FAIL: '{name}' not in DATASETS registry"
        print(f"  '{name}' registered: {DATASETS[name].__name__}")

    print("  PASS")
    return True


def test_eval_registry() -> bool:
    """Test that all new evaluations are in the registry."""
    print("\n=== Testing Eval Registry ===")
    from theseus.registry import EVALUATIONS

    expected = [
        "cfq",
        "clutrr",
        "longhealth",
        "mtob",
        "ccaligned_fr_ppl",
        "pes2o_ppl",
        "pg19_ppl",
        "pile_ppl",
        "mnli_ppl",
        "qqp_ppl",
        "sst2_ppl",
        "siqa_ppl",
        "winogrande_ppl",
        "fineweb_ppl",
    ]
    for name in expected:
        assert name in EVALUATIONS, f"  FAIL: '{name}' not in EVALUATIONS registry"
        print(f"  '{name}' registered")

    print("  PASS")
    return True


if __name__ == "__main__":
    results = {}

    # Always test registries first (no network needed)
    for test_fn in [test_registry, test_eval_registry]:
        name = test_fn.__name__
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    # Test indexed datasets (need network)
    for test_fn in [test_cfq, test_clutrr, test_longhealth, test_mtob]:
        name = test_fn.__name__
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    # Test streaming datasets (need network)
    for test_fn in [test_pg19, test_ccaligned, test_pes2o, test_pile]:
        name = test_fn.__name__
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{total} tests passed")

    if passed < total:
        sys.exit(1)
