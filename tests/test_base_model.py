import pytest
import tensorflow as tf
from Components.base_model import ModelBuilder
import numpy as np
from unittest.mock import patch, MagicMock



@pytest.fixture(autouse=True)
def disable_azure_auth(monkeypatch):
    """Prevents Azure ML auth during tests."""
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:/tmp/mlruns")  # local offline URI

@pytest.fixture
def model_builder_instance():
    """Fixture to create a ModelBuilder instance with default config."""
    return ModelBuilder(config_path="config/params.yaml")

def test_model_build_structure(model_builder_instance):
    """✅ Test that build_model returns valid Keras models."""
    base_model, model = model_builder_instance.build_model()
    assert base_model is not None, "Base model should not be None"
    assert model is not None, "Full model should not be None"
    assert len(model.layers) > 0, "Model should contain layers"

def test_model_compile_and_fit(model_builder_instance):
    """✅ Test that compile_and_fit runs one short training epoch."""
    # Build model
    base_model, model = model_builder_instance.build_model()

    # Create dummy dataset (small and fast)
    x = tf.random.normal((8, 224, 224, 3))
    y = tf.random.uniform((8,), maxval=3, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    # Call compile_and_fit (mock class weights)
    model, history = model_builder_instance.compile_and_fit(
        model=model,
        train_dataset=ds,
        val_dataset=ds,
        class_weights={0: 1.0, 1: 1.0, 2: 1.0},
    )
    print(model.metrics)
    # Check outputs
    assert model is not None, "Returned model should not be None"
    metric_names = []
    for m in model.metrics:
    # Direct metric (e.g., Mean, Accuracy)
        if hasattr(m, "name"):
            metric_names.append(m.name.lower())

        # CompileMetrics container (new in TF 2.15+/Keras 3.x)
        if hasattr(m, "metrics") and isinstance(m.metrics, (list, tuple)):
            for sub_m in m.metrics:
                if hasattr(sub_m, "name"):
                    metric_names.append(sub_m.name.lower())

    print("Extracted metric names:", metric_names)
    assert any("acc" in m for m in metric_names), f"Expected accuracy metric, found: {metric_names}"
    assert hasattr(history, "history"), "History object should have a history attribute"

def test_model_prediction(model_builder_instance):
    """✅ Test that model predicts on a sample batch."""
    _, model = model_builder_instance.build_model()
    sample_input = tf.random.normal((1, 224, 224, 3))
    preds = model(sample_input)
    assert preds.shape[-1] >= 1, "Prediction output should have class dimension"
    assert isinstance(preds.numpy(), (tf.Tensor, type(preds.numpy())))



@pytest.fixture
def dummy_model():
    """Simple TF model for testing evaluate/predict."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(4,)),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    return model


@pytest.fixture
def dummy_dataset():
    """Create a small dummy dataset (2 batches, labels 0–2)."""
    x = tf.random.normal((4, 4))
    y = tf.random.uniform((4,), maxval=3, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    return ds


@patch("mlflow.start_run")
@patch("mlflow.log_artifact")
@patch("mlflow.log_metric")
@patch("mlflow.set_experiment")
@patch("mlflow.keras.save_model")
@patch("mlflow.keras.log_model")
def test_test_model_success(
    mock_log_model,
    mock_save_model,
    mock_set_experiment,
    mock_log_metric,
    mock_log_artifact,
    mock_start_run,
    dummy_model,
    dummy_dataset
):
    """✅ Covers success path of test_model()."""
    mock_run = MagicMock()
    mock_run.info.run_id = "run_123"
    mock_start_run.return_value.__enter__.return_value = mock_run

    # ✅ Enforce dataset with correct 3-class labels
    x = tf.random.normal((4, 4))
    y = tf.constant([0, 1, 2, 1], dtype=tf.int32)
    dummy_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    dummy_model.evaluate = MagicMock(return_value=(0.5, 0.9))

    def fake_predict(batch):
        n = batch.shape[0]
        raw = np.random.rand(n, 3)
        return raw / raw.sum(axis=1, keepdims=True)

    dummy_model.predict = MagicMock(side_effect=fake_predict)

    mb = ModelBuilder()
    test_loss, test_acc, run_id = mb.test_model(dummy_model, dummy_dataset)

    assert test_loss == 0.5
    assert test_acc == 0.9
    assert run_id == "run_123"
    mock_save_model.assert_called_once()
    mock_log_metric.assert_any_call("test_loss", 0.5)
    mock_log_model.assert_called_once()

@patch("mlflow.keras.log_model")
@patch("mlflow.keras.save_model", side_effect=Exception("Save failed"))
def test_test_model_exception(mock_save, dummy_model, dummy_dataset):
    """✅ Covers the inner exception block (save_model failure)."""
    dummy_model.evaluate = MagicMock(return_value=(0.4, 0.85))

    # ✅ Force dataset labels to match model output (3 classes)
    x = tf.random.normal((4, 4))
    y = tf.constant([0, 1, 2, 1], dtype=tf.int32)
    dummy_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)

    def fake_predict(batch):
        n = batch.shape[0]
        raw = np.random.rand(n, 3)
        return raw / raw.sum(axis=1, keepdims=True)

    dummy_model.predict = MagicMock(side_effect=fake_predict)

    mb = ModelBuilder()
    result = mb.test_model(dummy_model, dummy_dataset)

    assert isinstance(result, tuple)
    assert len(result) == 3