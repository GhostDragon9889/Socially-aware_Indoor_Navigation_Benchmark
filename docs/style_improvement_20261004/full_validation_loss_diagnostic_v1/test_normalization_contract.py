import math
import struct
import unittest

from normalization_contract import EMA_RECIPE, fp32, validate_normalization


CONFIG = {"state_mean": [0.] * 5, "state_std": [5., 5., 1.5, 1.5, math.pi]}


def configured():
    return {key: [fp32(value) for value in values] for key, values in CONFIG.items()}


class NormalizationTests(unittest.TestCase):
    def test_raw_exact_and_only_supported_EMA_variants(self):
        raw = configured()
        result = validate_normalization(raw, CONFIG, 40000, weight_source="raw", ema_recipe=EMA_RECIPE)
        expected = {key: row["recipe_expected_fp32"] for key, row in result["buffers"].items()}
        self.assertEqual(expected["state_std"][2:4], [1.5000001192092896] * 2)
        self.assertEqual(result["EMA_blend_events"], 38000)
        for actual in (raw, expected):
            report = validate_normalization(actual, CONFIG, 40000, weight_source="EMA", ema_recipe=EMA_RECIPE)
            self.assertFalse(report["buffers_modified"])
        with self.assertRaises(ValueError):
            validate_normalization(expected, CONFIG, 40000, weight_source="raw", ema_recipe=EMA_RECIPE)

    def test_unsupported_ULP_shape_and_nonfinite_rejected(self):
        variants = []
        wrong = configured()
        wrong["state_mean"][0] = struct.unpack("<f", struct.pack("<I", 1))[0]
        variants.append(wrong)  # Zero must not accept an arbitrary subnormal ULP.
        wrong = configured()
        bits = struct.unpack("<I", struct.pack("<f", 5.))[0]
        wrong["state_std"][0] = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
        variants.append(wrong)  # Five does not drift in this registered recipe.
        variants += [{"state_mean": [0.] * 4, "state_std": [5.] * 5},
                     {"state_mean": [float("nan")] * 5, "state_std": [5.] * 5}]
        for actual in variants:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                validate_normalization(actual, CONFIG, 40000, weight_source="EMA", ema_recipe=EMA_RECIPE)


if __name__ == "__main__":
    unittest.main()
