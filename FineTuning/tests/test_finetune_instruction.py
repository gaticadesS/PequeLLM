from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn
from tokenizers import Tokenizer

FINETUNING_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FINETUNING_DIR.parent
sys.path.insert(0, str(FINETUNING_DIR))

from finetune_instruction import (  # noqa: E402
    IGNORE_INDEX,
    RESPONSE_MARKER,
    InstructionDataset,
    InstructionFineTuneConfig,
    collate_instruction_batch,
    evaluate_loss,
    format_input,
    format_response,
    generate_response,
    get_lr,
)

TOKENIZER = Tokenizer.from_file(str(REPO_ROOT / "tokenizer-culturax-es-hf.json"))
EOS_ID = TOKENIZER.token_to_id("</s>")
PAD_ID = TOKENIZER.token_to_id("<pad>")

SHORT = {"instruction": "Saluda.", "input": "", "output": "Hola."}
LONG = {"instruction": "Cuenta una historia.", "input": "", "output": "Había una vez un gato. " * 40}


class PromptFormatTests(unittest.TestCase):
    def test_generation_prompt_is_prefix_of_training_text(self):
        # Si entrenamiento e inferencia divergen en el marcador, el modelo recibe un prompt fuera de distribucion.
        entry = {"instruction": "Resume.", "input": "Un texto.", "output": "Listo."}
        self.assertTrue((format_input(entry) + format_response(entry)).startswith(format_input(entry) + RESPONSE_MARKER))


class DatasetTests(unittest.TestCase):
    def test_long_examples_are_dropped_by_default_and_all_end_with_eos(self):
        ds = InstructionDataset([SHORT, LONG], TOKENIZER, max_length=96, eos_token_id=EOS_ID)
        self.assertEqual(len(ds), 1)
        self.assertEqual(ds.num_dropped, 1)
        for token_ids, _ in ds.items:
            self.assertLessEqual(len(token_ids), 96)
            self.assertEqual(token_ids[-1], EOS_ID)

    def test_keep_truncated_trims_to_max_length(self):
        ds = InstructionDataset([SHORT, LONG], TOKENIZER, max_length=96, eos_token_id=EOS_ID, drop_too_long=False)
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.num_truncated, 1)
        self.assertEqual(max(len(ids) for ids, _ in ds.items), 96)

    def test_all_examples_too_long_raises(self):
        with self.assertRaises(ValueError):
            InstructionDataset([LONG], TOKENIZER, max_length=16, eos_token_id=EOS_ID)


class CollateTests(unittest.TestCase):
    def test_padding_is_ignored_in_targets(self):
        batch = [([5, 6, 7, 8], 2), ([5, 6], 1)]
        _, targets = collate_instruction_batch(batch, pad_token_id=PAD_ID, ignore_index=IGNORE_INDEX, mask_prompt_tokens=False)
        self.assertEqual(targets[1].tolist(), [6, IGNORE_INDEX, IGNORE_INDEX])

    def test_prompt_masking_hides_prompt_targets(self):
        batch = [([5, 6, 7, 8], 3)]
        _, targets = collate_instruction_batch(batch, pad_token_id=PAD_ID, ignore_index=IGNORE_INDEX, mask_prompt_tokens=True)
        self.assertEqual(targets[0].tolist(), [IGNORE_INDEX, IGNORE_INDEX, 8])


class LearningRateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = InstructionFineTuneConfig(lr=1e-3, warmup_steps=10, lr_min_ratio=0.1)

    def test_warmup_reaches_peak_then_cosine_decays_to_min(self):
        total = 100
        lrs = [get_lr(step, total, self.cfg) for step in range(total)]
        self.assertLess(lrs[0], self.cfg.lr)
        self.assertAlmostEqual(lrs[9], self.cfg.lr, places=9)
        self.assertTrue(all(a >= b for a, b in zip(lrs[10:], lrs[11:])))
        self.assertAlmostEqual(get_lr(total, total, self.cfg), self.cfg.lr * self.cfg.lr_min_ratio, places=9)

    def test_short_runs_still_reach_peak_lr(self):
        # warmup=100 por defecto, pero con 8 pasos totales se limita a total//2.
        cfg = InstructionFineTuneConfig(lr=1e-3, warmup_steps=100)
        self.assertAlmostEqual(max(get_lr(s, 8, cfg) for s in range(8)), cfg.lr, places=9)


class _ScriptedModel(nn.Module):
    """Modelo falso: emite los tokens de `script` en orden (uno por llamada)."""

    def __init__(self, script, vocab=65536):
        super().__init__()
        self.script = list(script)
        self.vocab = vocab
        self.calls = 0
        self.cfg = type("Cfg", (), {"block_size": 256})()

    def forward(self, idx, targets=None):
        logits = torch.zeros(idx.shape[0], idx.shape[1], self.vocab)
        logits[:, -1, self.script[min(self.calls, len(self.script) - 1)]] = 10.0
        self.calls += 1
        return logits, None


class GenerationTests(unittest.TestCase):
    def test_generation_stops_at_eos(self):
        word_ids = TOKENIZER.encode("hola mundo").ids
        model = _ScriptedModel(word_ids + [EOS_ID] + word_ids)
        response = generate_response(model, TOKENIZER, SHORT, device="cpu", max_new_tokens=50)
        self.assertEqual(model.calls, len(word_ids) + 1)
        self.assertEqual(response, TOKENIZER.decode(word_ids).strip())


class _FixedLossModel(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, idx, targets=None):
        return None, torch.tensor(self.loss)


class EvaluateLossTests(unittest.TestCase):
    def test_loss_is_weighted_by_non_ignored_tokens(self):
        amp = type("Amp", (), {"device_type": "cpu", "dtype": torch.float32, "enabled": False})()
        # batch1: 1 token valido; batch2: 3 tokens validos -> media ponderada = (2*1 + 6*3) / 4 = 5.0
        batches = [
            (torch.zeros(1, 3, dtype=torch.long), torch.tensor([[1, IGNORE_INDEX, IGNORE_INDEX]])),
            (torch.zeros(1, 3, dtype=torch.long), torch.tensor([[1, 1, 1]])),
        ]

        class _Model(_FixedLossModel):
            def __init__(self):
                super().__init__(0.0)
                self.values = iter([2.0, 6.0])

            def forward(self, idx, targets=None):
                return None, torch.tensor(next(self.values))

        self.assertAlmostEqual(evaluate_loss(_Model(), batches, "cpu", amp, max_batches=0), 5.0, places=6)


if __name__ == "__main__":
    unittest.main()
