import unittest
from unittest.mock import patch

from rt_rename.rename_service import rename_structures


class RenameServiceTests(unittest.TestCase):
    @patch("rt_rename.rename_service.generate_response")
    @patch("rt_rename.rename_service.read_guideline")
    def test_rename_structures_updates_rows_with_model_output(
        self,
        mock_read_guideline,
        mock_generate_response,
    ):
        mock_read_guideline.return_value = [{"name": "Heart"}, {"name": "Lung_L"}]
        mock_generate_response.return_value = {"response": "Heart,0.99"}

        rows = [{"local name": "hrt", "accept": False, "comment": ""}]
        updated_rows = rename_structures(
            model="Llama 3.1 | 70B | local",
            prompt="prompt_latest.txt",
            guideline="TG263",
            regions=["Thorax"],
            structure_dict=rows,
        )

        self.assertEqual(updated_rows[0]["TG263 name"], "Heart")
        self.assertEqual(updated_rows[0]["verify"], "pass")
        self.assertTrue(updated_rows[0]["accept"])
        self.assertNotEqual(updated_rows[0]["timestamp"], "")


if __name__ == "__main__":
    unittest.main()
