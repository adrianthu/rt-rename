import unittest
from unittest.mock import patch

from rt_rename.rename_service import rename_structures
from rt_rename.visual_context import StructureSliceImages


class FakeVisualContext:
    def get_slice_images(self, structure_name: str):
        if structure_name == "hrt":
            return StructureSliceImages(
                axial="data:image/png;base64,YXhpYWw=",
                sagittal="data:image/png;base64,c2FnaXR0YWw=",
                coronal="data:image/png;base64,Y29yb25hbA==",
            )
        return None


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

    @patch("rt_rename.rename_service.generate_response")
    @patch("rt_rename.rename_service.read_guideline")
    def test_rename_structures_adds_visual_context_for_image_models(
        self,
        mock_read_guideline,
        mock_generate_response,
    ):
        mock_read_guideline.return_value = [{"name": "Heart"}, {"name": "Lung_L"}]
        mock_generate_response.return_value = {"response": "Heart,High"}

        rename_structures(
            model="Gemma 3 | 27B | local",
            prompt="prompt_latest.txt",
            guideline="TG263",
            regions=["Thorax"],
            structure_dict=[{"local name": "hrt", "accept": False, "comment": ""}],
            visual_context=FakeVisualContext(),
        )

        request = mock_generate_response.call_args.args[0]
        self.assertEqual(request.model.display_name, "Gemma 3 | 27B | local")
        self.assertEqual(len(request.content_parts), 4)
        self.assertEqual(request.content_parts[1].type, "image_url")
        self.assertEqual(request.content_parts[1].image_url, "data:image/png;base64,YXhpYWw=")

    @patch("rt_rename.rename_service.read_guideline")
    def test_rename_structures_rejects_visual_context_for_text_only_models(
        self,
        mock_read_guideline,
    ):
        mock_read_guideline.return_value = [{"name": "Heart"}]

        with self.assertRaises(ValueError):
            rename_structures(
                model="Llama 3.1 | 70B | local",
                prompt="prompt_latest.txt",
                guideline="TG263",
                regions=["Thorax"],
                structure_dict=[{"local name": "hrt", "accept": False, "comment": ""}],
                visual_context=FakeVisualContext(),
            )


if __name__ == "__main__":
    unittest.main()
