import unittest

from rt_rename.inference import extract_prediction_and_confidence


class InferenceParsingTests(unittest.TestCase):
    def test_extract_prediction_and_confidence_uses_last_nonempty_line(self):
        response = "reasoning\n</think>\n\nHeart,0.91\n"
        prediction, confidence = extract_prediction_and_confidence(response)
        self.assertEqual(prediction, "Heart")
        self.assertEqual(confidence, "0.91")

    def test_extract_prediction_and_confidence_handles_missing_confidence(self):
        prediction, confidence = extract_prediction_and_confidence("Heart")
        self.assertEqual(prediction, "Heart")
        self.assertEqual(confidence, "")


if __name__ == "__main__":
    unittest.main()
