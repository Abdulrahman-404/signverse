import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

import test_normalizer
import test_matcher
import test_interpolator
import test_transitions
import test_smoother
import test_caching

test_normalizer.test_alef_normalization()
test_normalizer.test_yeh_normalization()
test_normalizer.test_teh_marbuta()
test_normalizer.test_kashida_removal()
test_normalizer.test_tashkeel_removal()
test_normalizer.test_normalize_search_key()
test_normalizer.test_remove_punctuation()
test_normalizer.test_is_arabic_word()
print("normalizer: OK")

test_matcher.test_trie_basic()
test_matcher.test_levenshtein()
test_matcher.test_norm_map_building()
print("matcher: OK")

test_interpolator.test_interpolate_same_length()
test_interpolator.test_interpolate_double()
test_interpolator.test_interpolate_single_frame()
test_interpolator.test_interpolate_empty()
print("interpolator: OK")

test_transitions.test_ease_in_out()
test_transitions.test_ease_in()
test_transitions.test_ease_out()
test_transitions.test_add_hold_frames()
test_transitions.test_crossfade()
test_transitions.test_apply_easing()
print("transitions: OK")

test_smoother.test_smoother_basic()
test_smoother.test_smoother_short_sequence()
test_smoother.test_smoother_fallback()
test_smoother.test_smoother_constant()
print("smoother: OK")

test_caching.test_cache_basic()
test_caching.test_cache_eviction()
test_caching.test_cache_file_load()
print("caching: OK")

print("\n" + "=" * 40)
print("  ALL TESTS PASSED")
print("=" * 40)
