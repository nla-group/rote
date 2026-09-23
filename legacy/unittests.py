import numpy as np
import pickle
import pytest
from slearn import symbolicML, slearn
from slearn.symbols import SAX, SAXTD, ESAX, MSAX, ASAX
from slearn.dmetric import (
    levenshtein_distance,
    hamming_distance,
    jaro_similarity,
    jaro_winkler_distance,
    cosine_similarity,
    cosine_bigram_similarity,
    lcs_distance,
    dice_coefficient,
    smith_waterman_distance,
    damerau_levenshtein_distance,
)


def run_sax_variant(model, ts, t, name, is_multivariate=False):
    """Run a SAX variant and compute RMSE between original and reconstructed time series."""
    symbols = model.fit_transform(ts)
    recon = model.inverse_transform()
    assert len(recon) == len(ts), f"{name} reconstruction length mismatch"
    rmse = np.sqrt(np.mean((ts - recon) ** 2))
    return rmse


def run_sax_variant_with_rmse(model, ts, t, name, is_multivariate=False):
    """Wrapper to run SAX variant and return RMSE."""
    rmse = run_sax_variant(model, ts, t, name, is_multivariate)
    return rmse


def run_standard_tests(t, ts, ts_multi, ts_constant, ts_short):
    """Run standard tests for SAX variants."""
    RMSE = []

    # Main test: standard time series
    sax = SAX(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(sax, ts, t, "SAX")
    RMSE.append(rmse)

    saxtd = SAXTD(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(saxtd, ts, t, "SAX-TD")
    RMSE.append(rmse)

    esax = ESAX(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(esax, ts, t, "eSAX")
    RMSE.append(rmse)

    msax = MSAX(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(msax, ts_multi, t, "mSAX", is_multivariate=True)
    RMSE.append(rmse)

    asax = ASAX(n_segments=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(asax, ts, t, "aSAX")
    RMSE.append(rmse)

    return RMSE


def run_edge_case_tests(t, ts_constant, ts_short):
    """Run edge case tests for SAX variants."""
    RMSE = []

    # Constant time series
    sax = SAX(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(sax, ts_constant, t, "SAX")
    RMSE.append(rmse)

    saxtd = SAXTD(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(saxtd, ts_constant, t, "SAX-TD")
    RMSE.append(rmse)

    esax = ESAX(window_size=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(esax, ts_constant, t, "eSAX")
    RMSE.append(rmse)

    asax = ASAX(n_segments=10, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(asax, ts_constant, t, "aSAX")
    RMSE.append(rmse)

    # Short time series
    sax = SAX(window_size=5, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(sax, ts_short, t[:20], "SAX")
    RMSE.append(rmse)

    saxtd = SAXTD(window_size=5, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(saxtd, ts_short, t[:20], "SAX-TD")
    RMSE.append(rmse)

    esax = ESAX(window_size=5, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(esax, ts_short, t[:20], "eSAX")
    RMSE.append(rmse)

    asax = ASAX(n_segments=5, alphabet_size=8)
    rmse = run_sax_variant_with_rmse(asax, ts_short, t[:20], "aSAX")
    RMSE.append(rmse)

    return RMSE


def save_results(RMSE, filename="test_rmse.pkl"):
    """Save RMSE values for later analysis."""
    with open(filename, "wb") as f:
        pickle.dump(RMSE, f)


@pytest.fixture
def setup_data():
    """Fixture to provide test data."""
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    ts = np.sin(t) + np.random.normal(0, 0.1, 100)  # Univariate, main test
    ts_multi = np.vstack([np.sin(t), np.cos(t)]).T + np.random.normal(
        0, 0.1, (100, 2)
    )  # Multivariate
    ts_constant = np.ones(100)  # Edge case: constant series
    t_short = np.linspace(0, 10, 20)
    ts_short = np.sin(t_short)  # Edge case: short series
    return t, ts, ts_multi, ts_constant, ts_short


@pytest.mark.parametrize(
    "rmse_threshold", [0.7], ids=lambda x: f"rmse_threshold={x}"
)
def test_sax_variants(setup_data, rmse_threshold, tmp_path):
    """Test SAX variants and verify RMSE values."""
    t, ts, ts_multi, ts_constant, ts_short = setup_data
    RMSE = []

    # Run standard tests
    RMSE.extend(run_standard_tests(t, ts, ts_multi, ts_constant, ts_short))

    # Run edge case tests
    RMSE.extend(run_edge_case_tests(t, ts_constant, ts_short))

    # Save results
    save_results(RMSE, tmp_path / "test_rmse.pkl")

    # Verify RMSE values are within acceptable bounds
    for i, rmse in enumerate(RMSE):
        assert rmse < rmse_threshold, f"RMSE {rmse} for test {i} exceeds threshold {rmse_threshold}"


def test_distances():
    """Test various distance and similarity metrics."""
    # Test Levenshtein
    assert levenshtein_distance("cat", "act") == 2
    assert levenshtein_distance("cat", "hat") == 1
    assert levenshtein_distance("cat", "cats") == 1
    assert levenshtein_distance("cat", "") == 3
    assert levenshtein_distance("kitten", "sitting") == 3

    # Test Hamming
    assert hamming_distance("karolin", "kathrin") == 3
    assert hamming_distance("10110", "11110") == 1
    with pytest.raises(ValueError):
        hamming_distance("cat", "cats")

    # Test Jaro
    assert pytest.approx(jaro_similarity("martha", "marhta"), 0.001) == 0.944
    assert jaro_similarity("cat", "") == 0.0
    assert jaro_similarity("same", "same") == 1.0

    # Test Jaro-Winkler
    assert pytest.approx(jaro_winkler_distance("martha", "marhta"), 0.001) == 0.961
    assert jaro_winkler_distance("dixon", "dicksonx") > 0.8
    assert jaro_winkler_distance("cat", "") == 0.0
    assert jaro_winkler_distance("same", "same") == 1.0
    assert jaro_winkler_distance("abc", "xyz") < 0.1

    # Test Cosine Similarity (Word-Based)
    assert pytest.approx(cosine_similarity("cat hat", "hat cat"), 0.001) == 1.0
    assert cosine_similarity("cat", "dog") == 0.0
    assert cosine_similarity("", "dog") == 0.0

    # Test Cosine Bigram Similarity
    assert pytest.approx(cosine_bigram_similarity("cat", "cap"), 0.001) == 0.5
    assert cosine_bigram_similarity("cat", "act") == 0.0
    assert cosine_bigram_similarity("cat", "dog") < 0.1
    assert cosine_bigram_similarity("", "dog") == 0.0

    # Test LCS Distance
    assert lcs_distance("kitten", "sitting") == 5
    assert lcs_distance("cat", "act") == 2
    assert lcs_distance("cat", "") == 3

    # Test Dice’s Coefficient
    assert pytest.approx(dice_coefficient("night", "nacht"), 0.001) == 0.25
    assert dice_coefficient("cat", "cat") == 1.0
    assert dice_coefficient("cat", "") == 0.0

    # Test Smith-Waterman
    assert smith_waterman_distance("kitten", "sitting") < 0
    assert smith_waterman_distance("cat", "act") < 0
    assert smith_waterman_distance("cat", "") == 0

    # Test Damerau-Levenshtein
    assert damerau_levenshtein_distance("cat", "act") == 1
    assert damerau_levenshtein_distance("cat", "hat") == 1
    assert damerau_levenshtein_distance("cat", "cats") == 1
    assert damerau_levenshtein_distance("cat", "") == 3
    assert damerau_levenshtein_distance("kitten", "sitting") == 3
