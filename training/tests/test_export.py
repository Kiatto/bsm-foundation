"""Tests for BLMF export and import."""

import json
import struct
import tempfile
from pathlib import Path

import torch
import pytest

from blm.model import BSMModel, BSMConfig
from blm.tokenizer import BPETokenizer
from blm.export import export_to_blmf, load_from_blmf, checksum64


class TestXXHash64:
    def test_known_empty(self):
        h = checksum64(b'')
        assert isinstance(h, int)
        assert h > 0

    def test_deterministic(self):
        data = b'test data for xxhash'
        assert checksum64(data) == checksum64(data)

    def test_different_inputs_different_hashes(self):
        assert checksum64(b'hello') != checksum64(b'world')

    def test_nonzero_output(self):
        assert checksum64(b'data') != 0


def make_tiny_model():
    """Create a minimal BSM model for testing."""
    cfg = BSMConfig(
        vocab_size=64,
        hidden_dim=64,
        num_layers=1,
        window_size=2,
        seq_len=8,
    )
    model = BSMModel(cfg)
    return model


def make_tiny_tokenizer():
    """Create a minimal tokenizer for testing."""
    tok = BPETokenizer(vocab_size=64)
    tok.train("the cat sat on the mat and the dog ran in the park")
    return tok


class TestExportToBLMF:
    def test_export_creates_file(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            size = export_to_blmf(model, tok, out_path)
            assert size > 0, "File should have content"
            assert Path(out_path).stat().st_size == size
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_export_file_starts_with_magic(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            export_to_blmf(model, tok, out_path)
            with open(out_path, 'rb') as f:
                magic = f.read(4)
            assert magic == b'BLMF', f"Expected BLMF magic, got {magic!r}"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_load_returns_header(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            export_to_blmf(model, tok, out_path)
            result = load_from_blmf(out_path)
            assert "header" in result
            assert result["header"]["arch"] == "BSM"
            assert result["header"]["vocab_size"] == 64
            assert result["header"]["hidden_dim"] == 64
            assert result["header"]["num_layers"] == 1
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_load_returns_sections(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            export_to_blmf(model, tok, out_path)
            result = load_from_blmf(out_path)
            assert "sections" in result

            sec_names = list(result["sections"].keys())
            assert "embedding" in sec_names, f"Missing embedding section: {sec_names}"
            assert "layer_0_wforget" in sec_names
            assert "layer_0_winput" in sec_names
            assert "layer_0_wmix" in sec_names
            assert "head_weight" in sec_names
            assert "vocab" in sec_names
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_export_with_metadata(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            export_to_blmf(model, tok, out_path, metadata={
                "description": "test model",
                "training_steps": 100,
            })
            result = load_from_blmf(out_path)
            assert result["header"]["description"] == "test model"
            assert result["header"]["training_steps"] == 100
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_checksum_valid(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            export_to_blmf(model, tok, out_path)
            with open(out_path, 'rb') as f:
                data = f.read()
            # Checksum is last 8 bytes, computed over everything before it
            stored_checksum = struct.unpack_from('<Q', data, len(data) - 8)[0]
            computed = checksum64(data[:-8])
            assert stored_checksum == computed, "Checksum mismatch"
        finally:
            Path(out_path).unlink(missing_ok=True)

    def test_section_data_sizes(self):
        model = make_tiny_model()
        tok = make_tiny_tokenizer()

        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            out_path = f.name

        try:
            export_to_blmf(model, tok, out_path)
            result = load_from_blmf(out_path)

            emb = result["sections"]["embedding"]
            # vocab=64, hidden=64 → 64 * 8 = 512 bytes
            assert emb["shape"] == (64, 8), f"Unexpected embedding shape: {emb['shape']}"

            hw = result["sections"]["head_weight"]
            # [vocab_size, hidden_dim] float32
            assert hw["dtype"] == 3  # float32
        finally:
            Path(out_path).unlink(missing_ok=True)


class TestExportErrors:
    def test_invalid_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix='.blmf', delete=False) as f:
            f.write(b'not a blmf file')
            f_path = f.name

        try:
            with pytest.raises(ValueError, match="Not a BLMF file"):
                load_from_blmf(f_path)
        finally:
            Path(f_path).unlink(missing_ok=True)
