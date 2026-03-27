import base64
import unittest

from rt_rename.parsers import parse_csv, parse_filenames


class ParserTests(unittest.TestCase):
    def test_parse_filenames_filters_auxiliary_and_target_volume_files(self):
        rows = parse_filenames(
            [
                "PTV_7000.nrrd",
                "Lung_L.nrrd",
                "Kidney.nrrd",
                "Body_stitched.nrrd",
                "Body_s2.nrrd",
            ],
            tv_filter=True,
        )
        self.assertEqual([row["local name"] for row in rows], ["Kidney", "Lung_L"])

    def test_parse_csv_preserves_existing_columns(self):
        csv_payload = (
            "local name,TG263 name,confidence,verify,accept,comment,raw output,timestamp\n"
            "Heart.nrrd,Heart,0.98,pass,True,ready,model output,2026-01-01T00:00:00Z\n"
        )
        encoded = base64.b64encode(csv_payload.encode("utf-8")).decode("utf-8")
        rows = parse_csv(f"data:text/csv;base64,{encoded}", "structures.csv")
        self.assertEqual(rows[0]["local name"], "Heart")
        self.assertEqual(rows[0]["TG263 name"], "Heart")
        self.assertTrue(rows[0]["accept"])
        self.assertEqual(rows[0]["raw output"], "model output")


if __name__ == "__main__":
    unittest.main()
