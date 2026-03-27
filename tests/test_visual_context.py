import base64
from io import BytesIO
import unittest

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, RTStructureSetStorage, generate_uid

from rt_rename.visual_context import StructureImageContext


def _dataset_to_upload_contents(dataset: Dataset) -> str:
    buffer = BytesIO()
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:application/octet-stream;base64,{encoded}"


def _file_meta(sop_class_uid: str, sop_instance_uid: str) -> FileMetaDataset:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.ImplementationClassUID = generate_uid()
    return file_meta


def _make_ct_slice(z_index: int, sop_instance_uid: str) -> Dataset:
    dataset = Dataset()
    dataset.file_meta = _file_meta(CTImageStorage, sop_instance_uid)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.100"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.200"
    dataset.Modality = "CT"
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = "123"
    dataset.Rows = 10
    dataset.Columns = 10
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelRepresentation = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.ImagePositionPatient = [0.0, 0.0, float(z_index)]
    dataset.InstanceNumber = z_index + 1
    pixels = np.arange(100, dtype=np.int16).reshape(10, 10) + z_index * 25
    dataset.PixelData = pixels.tobytes()
    return dataset


def _make_rtstruct(referenced_uid: str) -> Dataset:
    dataset = Dataset()
    sop_instance_uid = generate_uid()
    dataset.file_meta = _file_meta(RTStructureSetStorage, sop_instance_uid)
    dataset.SOPClassUID = RTStructureSetStorage
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.100"
    dataset.SeriesInstanceUID = "1.2.826.0.1.3680043.8.498.300"
    dataset.Modality = "RTSTRUCT"
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = "123"
    dataset.StructureSetLabel = "TEST"

    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = "Heart"
    dataset.StructureSetROISequence = [roi]

    contour_image = Dataset()
    contour_image.ReferencedSOPInstanceUID = referenced_uid

    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = 4
    contour.ContourImageSequence = [contour_image]
    contour.ContourData = [
        2.0, 3.0, 2.0,
        6.0, 3.0, 2.0,
        6.0, 7.0, 2.0,
        2.0, 7.0, 2.0,
    ]

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ContourSequence = [contour]
    dataset.ROIContourSequence = [roi_contour]
    return dataset


class VisualContextTests(unittest.TestCase):
    def test_structure_image_context_returns_three_slice_images(self):
        ct_uids = [generate_uid() for _ in range(5)]
        ct_uploads = [
            _dataset_to_upload_contents(_make_ct_slice(index, uid))
            for index, uid in enumerate(ct_uids)
        ]
        rtstruct_upload = _dataset_to_upload_contents(_make_rtstruct(ct_uids[2]))

        context = StructureImageContext.from_uploads(rtstruct_upload, ct_uploads)
        slice_images = context.get_slice_images("Heart")

        self.assertIsNotNone(slice_images)
        self.assertTrue(slice_images.axial.startswith("data:image/png;base64,"))
        self.assertTrue(slice_images.sagittal.startswith("data:image/png;base64,"))
        self.assertTrue(slice_images.coronal.startswith("data:image/png;base64,"))

    def test_structure_image_context_returns_none_for_missing_structure(self):
        ct_uids = [generate_uid() for _ in range(3)]
        ct_uploads = [
            _dataset_to_upload_contents(_make_ct_slice(index, uid))
            for index, uid in enumerate(ct_uids)
        ]
        rtstruct_upload = _dataset_to_upload_contents(_make_rtstruct(ct_uids[1]))

        context = StructureImageContext.from_uploads(rtstruct_upload, ct_uploads)
        self.assertIsNone(context.get_slice_images("Unknown"))


if __name__ == "__main__":
    unittest.main()
