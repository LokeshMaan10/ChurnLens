"""Tests for the hash-based user sampling rule."""

import hashlib

import pyarrow as pa

from churnlens.data.sampling import UserSampler, keep_user

SEED = 42
PCT = 10


def test_keep_user_matches_spec_formula() -> None:
    msno = "SyntheticTestUserId1111111111111111111111BBB="   # made up, same format
    expected = int(hashlib.sha256(f"{msno}{SEED}".encode()).hexdigest(), 16) % 100 < PCT
    assert keep_user(msno, SEED, PCT) is expected


def test_sampling_is_deterministic_and_seed_dependent() -> None:
    ids = [f"user{i}" for i in range(2000)]
    first = [keep_user(m, SEED, PCT) for m in ids]
    assert first == [keep_user(m, SEED, PCT) for m in ids]
    assert first != [keep_user(m, SEED + 1, PCT) for m in ids]


def test_sample_rate_close_to_pct() -> None:
    ids = [f"user{i}" for i in range(20000)]
    rate = sum(keep_user(m, SEED, PCT) for m in ids) / len(ids)
    assert abs(rate - PCT / 100) < 0.01


def test_mask_agrees_with_scalar_rule_and_is_chunk_invariant() -> None:
    ids = [f"user{i % 700}" for i in range(3000)]  # repeated ids, like log rows
    sampler = UserSampler(SEED, PCT)
    whole = sampler.mask(pa.array(ids)).to_pylist()
    assert whole == [keep_user(m, SEED, PCT) for m in ids]
    halves = UserSampler(SEED, PCT)
    split = (halves.mask(pa.array(ids[:1500])).to_pylist()
             + halves.mask(pa.array(ids[1500:])).to_pylist())
    assert split == whole


def test_full_mode_keeps_everyone() -> None:
    assert all(UserSampler(SEED, 100).mask(pa.array(["a", "b", "c"])).to_pylist())
