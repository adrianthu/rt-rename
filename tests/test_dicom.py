import unittest

from pydicom.dataset import Dataset

from rt_rename.dicom_utils import is_ct_image_dataset, update_dicom


class DicomUpdateTests(unittest.TestCase):
    def test_update_dicom_changes_only_accepted_structures(self):
        dataset = Dataset()
        roi_one = Dataset()
        roi_one.ROIName = "Heart"
        roi_two = Dataset()
        roi_two.ROIName = "Lung_L"
        dataset.StructureSetROISequence = [roi_one, roi_two]

        updated = update_dicom(
            dataset,
            [
                {"local name": "Heart", "TG263 name": "Heart_PRV", "accept": True},
                {"local name": "Lung_L", "TG263 name": "Lung_L", "accept": False},
            ],
        )

        self.assertEqual(updated.StructureSetROISequence[0].ROIName, "Heart_PRV")
        self.assertEqual(updated.StructureSetROISequence[1].ROIName, "Lung_L")

    def test_is_ct_image_dataset_requires_ct_modality_and_pixels(self):
        dataset = Dataset()
        dataset.Modality = "CT"
        dataset.PixelData = b"1234"

        self.assertTrue(is_ct_image_dataset(dataset))

        dataset.Modality = "MR"
        self.assertFalse(is_ct_image_dataset(dataset))


if __name__ == "__main__":
    unittest.main()
