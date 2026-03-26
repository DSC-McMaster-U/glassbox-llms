"""
Tests for TransformersModelWrapper with model_class parameter.

Uses mocks to avoid downloading real models in CI.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch


def _make_mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    return tok


def _make_mock_hf_model(**extra_attrs):
    model = MagicMock()
    model.named_modules.return_value = []
    for k, v in extra_attrs.items():
        setattr(model, k, v)
    return model


class TestModelClassParameter:
    """Tests for the model_class parameter on TransformersModelWrapper."""

    @patch("glassboxllms.models.huggingface.AutoTokenizer")
    @patch("glassboxllms.models.huggingface.AutoModel")
    def test_default_model_class_uses_auto_model(self, mock_auto_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        mock_auto_cls.from_pretrained.return_value = _make_mock_hf_model()

        from glassboxllms.models.huggingface import TransformersModelWrapper, _MODEL_CLASSES

        # Temporarily override the dict so the class under test calls our mock
        original = _MODEL_CLASSES["auto"]
        _MODEL_CLASSES["auto"] = mock_auto_cls
        try:
            wrapper = TransformersModelWrapper("gpt2")
            mock_auto_cls.from_pretrained.assert_called_once_with("gpt2")
            assert wrapper._model_class == "auto"
        finally:
            _MODEL_CLASSES["auto"] = original

    @patch("glassboxllms.models.huggingface.AutoTokenizer")
    @patch("glassboxllms.models.huggingface.AutoModelForCausalLM")
    def test_causal_lm_class_uses_causal_model(self, mock_causal_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        mock_causal_cls.from_pretrained.return_value = _make_mock_hf_model()

        from glassboxllms.models.huggingface import TransformersModelWrapper, _MODEL_CLASSES

        original = _MODEL_CLASSES["causal_lm"]
        _MODEL_CLASSES["causal_lm"] = mock_causal_cls
        try:
            wrapper = TransformersModelWrapper("gpt2", model_class="causal_lm")
            mock_causal_cls.from_pretrained.assert_called_once_with("gpt2")
            assert wrapper._model_class == "causal_lm"
        finally:
            _MODEL_CLASSES["causal_lm"] = original

    def test_invalid_model_class_raises(self):
        from glassboxllms.models.huggingface import TransformersModelWrapper

        with pytest.raises(ValueError, match="Unknown model_class"):
            TransformersModelWrapper("gpt2", model_class="encoder_decoder")

    @patch("glassboxllms.models.huggingface.AutoTokenizer")
    @patch("glassboxllms.models.huggingface.AutoModelForCausalLM")
    def test_lm_head_property_accessible_for_causal_lm(self, mock_causal_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        hf_model = _make_mock_hf_model(lm_head=nn.Linear(10, 100))
        mock_causal_cls.from_pretrained.return_value = hf_model

        from glassboxllms.models.huggingface import TransformersModelWrapper, _MODEL_CLASSES

        original = _MODEL_CLASSES["causal_lm"]
        _MODEL_CLASSES["causal_lm"] = mock_causal_cls
        try:
            wrapper = TransformersModelWrapper("gpt2", model_class="causal_lm")
            assert wrapper.lm_head is not None
        finally:
            _MODEL_CLASSES["causal_lm"] = original

    @patch("glassboxllms.models.huggingface.AutoTokenizer")
    @patch("glassboxllms.models.huggingface.AutoModel")
    def test_lm_head_property_raises_for_auto_model(self, mock_auto_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        # MagicMock auto-creates attributes, so use spec to restrict
        hf_model = MagicMock(spec=["named_modules", "eval", "to"])
        hf_model.named_modules.return_value = []
        mock_auto_cls.from_pretrained.return_value = hf_model

        from glassboxllms.models.huggingface import TransformersModelWrapper, _MODEL_CLASSES

        original = _MODEL_CLASSES["auto"]
        _MODEL_CLASSES["auto"] = mock_auto_cls
        try:
            wrapper = TransformersModelWrapper("gpt2")
            with pytest.raises(AttributeError, match="lm_head"):
                _ = wrapper.lm_head
        finally:
            _MODEL_CLASSES["auto"] = original

    @patch("glassboxllms.models.huggingface.AutoTokenizer")
    @patch("glassboxllms.models.huggingface.AutoModelForCausalLM")
    def test_repr_includes_model_class(self, mock_causal_cls, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _make_mock_tokenizer()
        mock_causal_cls.from_pretrained.return_value = _make_mock_hf_model()

        from glassboxllms.models.huggingface import TransformersModelWrapper, _MODEL_CLASSES

        original = _MODEL_CLASSES["causal_lm"]
        _MODEL_CLASSES["causal_lm"] = mock_causal_cls
        try:
            wrapper = TransformersModelWrapper("gpt2", model_class="causal_lm")
            r = repr(wrapper)
            assert "causal_lm" in r
            assert "gpt2" in r
        finally:
            _MODEL_CLASSES["causal_lm"] = original
