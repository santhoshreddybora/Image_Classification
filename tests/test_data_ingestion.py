import os
import pytest
from unittest.mock import patch, MagicMock
from Components.data_ingestion import (
    DataIngestion,
)  # adjust import to match repo structure
from entity.artifact_entity import DataIngestionArtifact


@pytest.fixture
def mock_params(monkeypatch, tmp_path):
    """Mock params for data ingestion"""
    fake_params = {
        "container_name": "test-container",
        "blob_name": "test.zip",
        "data_zip_file_path": str(tmp_path / "zip"),
        "data_unzip_path": str(tmp_path / "unzipped"),
    }
    monkeypatch.setattr("Components.data_ingestion.params", fake_params)
    return fake_params


@patch("Components.data_ingestion.BlobServiceClient")
@patch("Components.data_ingestion.conn_str", "fake_connection")
def test_download_and_extract_data(mock_blob_service, mock_params):
    """Simulate downloading from Azure Blob Storage"""
    fake_blob = MagicMock()
    fake_blob.download_blob.return_value.readall.return_value = b"FakeZipData"
    fake_container = MagicMock()
    fake_container.list_blobs.return_value = [MagicMock(name="test.zip")]

    mock_blob_service.from_connection_string.return_value.get_blob_client.return_value = (
        fake_blob
    )
    mock_blob_service.from_connection_string.return_value.get_container_client.return_value = (
        fake_container
    )

    ingestion = DataIngestion()
    ingestion.download_and_extract_data()

    assert os.path.exists(mock_params["data_zip_file_path"])


def test_unzip_data(tmp_path, monkeypatch):
    """Ensure unzip_data extracts correctly"""
    import zipfile

    zip_dir = tmp_path / "zip"
    unzip_dir = tmp_path / "unzipped"
    os.makedirs(zip_dir, exist_ok=True)
    zip_path = zip_dir / "test.zip"

    # Create fake zip
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("test.txt", "dummy")

    monkeypatch.setattr(
        "Components.data_ingestion.params",
        {
            "data_zip_file_path": str(zip_dir),
            "data_unzip_path": str(unzip_dir),
            "blob_name": "test.zip",
        },
    )

    ingestion = DataIngestion()
    ingestion.unzip_data()

    assert os.path.exists(unzip_dir / "test.txt")


def test_initiate_data_ingestion(monkeypatch):
    """Test that initiate_data_ingestion returns an artifact"""
    ingestion = DataIngestion()
    monkeypatch.setattr(ingestion, "download_and_extract_data", lambda: None)
    monkeypatch.setattr(ingestion, "unzip_data", lambda: None)
    artifact = ingestion.initiate_data_ingestion()
    assert isinstance(artifact, DataIngestionArtifact)
