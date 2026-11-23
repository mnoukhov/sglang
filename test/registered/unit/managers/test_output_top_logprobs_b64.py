"""Unit tests for dense generated-token top logprobs.

Covers dense decode top-k extraction, numpy row transport from the scheduler,
and the base64 response format (`return_output_top_logprobs_b64`).
"""

import asyncio
import base64
import pickle
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.logprob_processor import (
    OutputLogprobProcessor,
    get_top_logprobs,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler_components import output_streamer
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    TokenizerManager,
    _build_output_top_logprobs_b64_fields,
    _dense_output_top_logprobs_chunk,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _TokenizerManagerStub:
    convert_logprob_style = TokenizerManager.convert_logprob_style
    add_logprob_to_meta_info = TokenizerManager.add_logprob_to_meta_info
    detokenize_logprob_tokens = TokenizerManager.detokenize_logprob_tokens
    detokenize_top_logprobs_tokens = TokenizerManager.detokenize_top_logprobs_tokens


def _make_state(**obj_kwargs) -> ReqState:
    obj = GenerateReqInput(text="hello", **obj_kwargs)
    obj.normalize_batch_and_arguments()
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _recv(values, indices, finished):
    return SimpleNamespace(
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=[[-0.5] * len(values)],
        output_token_logprobs_idx=[[1] * len(values)],
        input_top_logprobs_val=None,
        input_top_logprobs_val_flat=None,
        output_top_logprobs_val=[values],
        output_top_logprobs_idx=[indices],
        input_token_ids_logprobs_val=None,
        output_token_ids_logprobs_val=None,
        finished_reasons=[{"type": "length"} if finished else None],
    )


def _convert(state, recv_obj, k):
    meta_info = {}
    _TokenizerManagerStub().convert_logprob_style(
        meta_info,
        state,
        top_logprobs_num=k,
        token_ids_logprob=None,
        return_text_in_logprobs=False,
        recv_obj=recv_obj,
        recv_obj_index=0,
    )
    return meta_info


def _decode(meta_info):
    rows, k = meta_info["output_top_logprobs_shape"]
    values = np.frombuffer(
        base64.b64decode(meta_info["output_top_logprobs_val_b64"]), dtype="<f4"
    ).reshape(rows, k)
    indices = np.frombuffer(
        base64.b64decode(meta_info["output_top_logprobs_idx_b64"]), dtype="<i4"
    ).reshape(rows, k)
    return values, indices


# Float32-exact values so the nested and dense paths compare exactly.
_VALUES = np.array([[-0.5, -1.25], [-0.125, -2.5], [-0.25, -3.0]], dtype=np.float32)
_INDICES = np.array([[7, 9], [3, 4], [11, 2]], dtype=np.int64)


class TestRequestValidation(CustomTestCase):
    def test_default_off(self):
        obj = GenerateReqInput(text="hello")
        obj.normalize_batch_and_arguments()
        self.assertFalse(obj.return_output_top_logprobs_b64)

    def test_rejects_streaming(self):
        obj = GenerateReqInput(
            text="hello", stream=True, return_output_top_logprobs_b64=True
        )
        with self.assertRaises(ValueError):
            obj.normalize_batch_and_arguments()

    def test_propagates_to_batch_items(self):
        obj = GenerateReqInput(
            text=["a", "b"],
            return_logprob=True,
            top_logprobs_num=2,
            return_output_top_logprobs_b64=True,
        )
        obj.normalize_batch_and_arguments()
        self.assertTrue(obj[1].return_output_top_logprobs_b64)


class TestDenseDecodeTopk(CustomTestCase):
    def test_uniform_batch_returns_dense_matrices(self):
        torch.manual_seed(0)
        logprobs = torch.log_softmax(torch.randn(4, 50), dim=-1)
        expected_values, expected_indices = get_top_logprobs(
            logprobs.clone(), [3] * 4, no_copy_to_cpu=True
        )
        result = OutputLogprobProcessor().compute_logprobs(
            logprobs, [3] * 4, [None] * 4, torch.zeros(4, dtype=torch.long)
        )
        self.assertIsInstance(result.top_logprobs_val, torch.Tensor)
        self.assertEqual(tuple(result.top_logprobs_val.shape), (4, 3))
        for row in range(4):
            torch.testing.assert_close(
                result.top_logprobs_val[row], expected_values[row], rtol=0, atol=0
            )
            torch.testing.assert_close(
                result.top_logprobs_idx[row], expected_indices[row], rtol=0, atol=0
            )

    def test_mixed_widths_keep_row_lists(self):
        logprobs = torch.log_softmax(torch.randn(3, 20), dim=-1)
        result = OutputLogprobProcessor().compute_logprobs(
            logprobs, [2, 0, 3], [None] * 3, torch.zeros(3, dtype=torch.long)
        )
        self.assertIsInstance(result.top_logprobs_val, list)
        self.assertEqual([len(row) for row in result.top_logprobs_val], [2, 0, 3])

    def test_finalize_dense_rows_are_numpy_views(self):
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_top_logprobs_val=torch.from_numpy(_VALUES.copy()),
            next_token_top_logprobs_idx=torch.from_numpy(_INDICES.copy()),
        )
        output.finalize_output_top_logprobs()
        self.assertIsInstance(output.next_token_top_logprobs_val, np.ndarray)
        np.testing.assert_array_equal(output.next_token_top_logprobs_val, _VALUES)
        self.assertEqual(output.next_token_top_logprobs_idx[1].tolist(), [3, 4])

    def test_finalize_row_lists_unchanged(self):
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_top_logprobs_val=[torch.tensor([-0.5]), torch.tensor([])],
            next_token_top_logprobs_idx=[torch.tensor([4]), torch.tensor([])],
        )
        output.finalize_output_top_logprobs()
        self.assertEqual(output.next_token_top_logprobs_val, [[-0.5], []])
        self.assertEqual(output.next_token_top_logprobs_idx, [[4], []])


class TestStreamerPacking(CustomTestCase):
    def test_numpy_rows_pack_into_one_array(self):
        rows = [_VALUES[i] for i in range(3)]
        packed = output_streamer._pack_top_logprob_rows(rows)
        self.assertIsInstance(packed, np.ndarray)
        self.assertEqual(packed.dtype, np.float32)
        np.testing.assert_array_equal(packed, _VALUES)
        # Survives the default pickle IPC as a single buffer.
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(packed)), _VALUES)

    def test_prefill_list_row_then_numpy_rows(self):
        rows = [_VALUES[0].tolist(), _VALUES[1], _VALUES[2]]
        np.testing.assert_array_equal(
            output_streamer._pack_top_logprob_rows(rows), _VALUES
        )

    def test_list_rows_unchanged(self):
        rows = [[-0.5, -1.0], [-0.25, -2.0]]
        self.assertIs(output_streamer._pack_top_logprob_rows(rows), rows)
        self.assertEqual(output_streamer._pack_top_logprob_rows([]), [])

    def test_ragged_rows_fall_back_to_lists(self):
        rows = [[], np.array([-0.5, -1.0], dtype=np.float32)]
        self.assertEqual(
            output_streamer._pack_top_logprob_rows(rows), [[], [-0.5, -1.0]]
        )

    def test_msgpack_ipc_sends_lists(self):
        original = output_streamer._USE_PICKLE_IPC
        output_streamer._USE_PICKLE_IPC = False
        try:
            packed = output_streamer._pack_top_logprob_rows([_VALUES[0], _VALUES[1]])
        finally:
            output_streamer._USE_PICKLE_IPC = original
        self.assertEqual(packed, _VALUES[:2].tolist())


class TestNestedResponseFromDenseChunks(CustomTestCase):
    def test_dense_chunk_matches_list_chunk(self):
        dense_state = _make_state(return_logprob=True, top_logprobs_num=2)
        list_state = _make_state(return_logprob=True, top_logprobs_num=2)
        dense = _convert(dense_state, _recv(_VALUES, _INDICES, True), 2)
        nested = _convert(
            list_state, _recv(_VALUES.tolist(), _INDICES.tolist(), True), 2
        )
        self.assertEqual(dense["output_top_logprobs"], nested["output_top_logprobs"])
        self.assertIsInstance(dense["output_top_logprobs"][0][0][0], float)
        self.assertIsInstance(dense["output_top_logprobs"][0][0][1], int)
        self.assertNotIn("output_top_logprobs_val_b64", dense)


class TestB64Response(CustomTestCase):
    def _state(self, k=2):
        return _make_state(
            return_logprob=True, top_logprobs_num=k, return_output_top_logprobs_b64=True
        )

    def test_fields_only_at_finish_and_chunks_concatenate(self):
        state = self._state()
        first = _convert(state, _recv(_VALUES[:2], _INDICES[:2], False), 2)
        self.assertNotIn("output_top_logprobs_val_b64", first)
        self.assertNotIn("output_top_logprobs", first)
        last = _convert(state, _recv(_VALUES[2:], _INDICES[2:], True), 2)
        values, indices = _decode(last)
        np.testing.assert_array_equal(values, _VALUES)
        np.testing.assert_array_equal(indices, _INDICES)
        self.assertEqual(last["output_top_logprobs_shape"], [3, 2])
        self.assertEqual(last["output_top_logprobs_val_b64_dtype"], "float32")
        self.assertEqual(last["output_top_logprobs_idx_b64_dtype"], "int32")
        self.assertNotIn("output_top_logprobs", last)

    def test_matches_nested_values(self):
        nested_state = _make_state(return_logprob=True, top_logprobs_num=2)
        nested = _convert(nested_state, _recv(_VALUES, _INDICES, True), 2)
        values, indices = _decode(
            _convert(self._state(), _recv(_VALUES, _INDICES, True), 2)
        )
        for row, entries in enumerate(nested["output_top_logprobs"]):
            self.assertEqual([entry[0] for entry in entries], values[row].tolist())
            self.assertEqual([entry[1] for entry in entries], indices[row].tolist())

    def test_list_chunks_with_missing_rows_are_padded(self):
        state = self._state(k=3)
        meta = _convert(
            state,
            _recv([None, [-0.5], [-0.25, -1.0, -2.0]], [None, [4], [1, 2, 3]], True),
            3,
        )
        values, indices = _decode(meta)
        self.assertTrue(np.isneginf(values[0]).all())
        self.assertEqual(indices[0].tolist(), [-1, -1, -1])
        self.assertEqual(values[1].tolist()[:1], [-0.5])
        self.assertEqual(indices[1].tolist(), [4, -1, -1])
        self.assertEqual(indices[2].tolist(), [1, 2, 3])

    def test_no_rows(self):
        fields = _build_output_top_logprobs_b64_fields([], 4)
        self.assertEqual(fields["output_top_logprobs_shape"], [0, 4])
        self.assertEqual(fields["output_top_logprobs_val_b64"], "")

    def test_dense_chunk_casts_indices(self):
        values, indices = _dense_output_top_logprobs_chunk(_VALUES, _INDICES, 2)
        self.assertEqual(values.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int32)
        np.testing.assert_array_equal(indices, _INDICES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
