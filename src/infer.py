#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot mutational effect inference with Ankh.

Reads ProteinGym substitution-style CSVs and computes a masked-marginal delta
log-probability score per variant:
  Δ = log P(mutant_aa | wildtype context) - log P(wildtype_aa | context)

Expected input columns:
  - `mut_info`: like "A123G" (WT AA, 1-based pos, MUT AA). Multi-muts may be
    separated by ":" / ";" / "," and are scored as a sum of per-site deltas.
  - `seq`: full mutant sequence
  - `fitness`: experimental score
"""

import argparse
import os
import re
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import login
from scipy.stats import spearmanr

import ankh

DEFAULT_DATA_DIR = Path("/opt/ml/processing/input/data")
DEFAULT_OUTPUT_DIR = Path("/opt/ml/processing/output")

REQUIRED_COLS = {"seq", "fitness", "mut_info"}
AA20 = "ACDEFGHIKLMNPQRSTVWY"


_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=None, help="Hugging Face token (or env HF_TOKEN / HUGGINGFACE_HUB_TOKEN).")
    parser.add_argument("--input_csv", default=None, help="Only process this CSV (basename under data_dir, or absolute path).")
    parser.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output_suffix", default="_ankh_zeroshot.csv")
    parser.add_argument("--model_name", default="ankh_base", choices=["ankh_base", "ankh_large", "ankh3_large", "ankh3_xl"])
    parser.add_argument("--batch_size", type=int, default=1024, help="Shard size for per-position masking forward passes.")
    parser.add_argument("--progress_every", type=int, default=0, help="Print progress every N variants (0=off).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    else:
        print("Warning: no hf_token provided; proceeding without Hugging Face login.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Ankh model {args.model_name} (device={device})")
    model, tokenizer = ankh.load_model(args.model_name, generation=True)
    model.to(device).eval()

    aa_to_id: dict[str, int] = {}
    for aa in AA20:
        ids = tokenizer(
            [[aa]],
            add_special_tokens=False,
            is_split_into_words=True,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        if len(ids) != 1:
            raise RuntimeError(f"Unexpected tokenization for '{aa}': {ids}")
        aa_to_id[aa] = ids[0]

    if args.model_name.startswith("ankh3_"):
        prefix, shift = "[NLU]", 1
    else:
        prefix, shift = None, 0

    @torch.no_grad()
    def masked_log_probs_all_positions(wt_seq: str) -> torch.Tensor:
        extra_id_0 = tokenizer.get_vocab()["<extra_id_0>"]
        aa_tokens = ([prefix] if prefix else []) + list(wt_seq)

        enc = tokenizer(
            [aa_tokens],
            add_special_tokens=True,
            is_split_into_words=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        seqlen = len(wt_seq)
        expanded_input_ids = input_ids.expand((seqlen, -1))
        expanded_attention_mask = attention_mask.expand((seqlen, -1))

        idxs = torch.arange(shift, seqlen + shift, device=device).view(-1, 1)
        mask_tensor = torch.full(
            (seqlen, 1),
            extra_id_0,
            device=device,
            dtype=expanded_input_ids.dtype,
        )
        masked_input_ids = torch.scatter(expanded_input_ids, dim=-1, index=idxs, src=mask_tensor)

        decoder_input_ids = torch.tensor(
            [model.config.decoder_start_token_id, extra_id_0],
            device=device,
            dtype=masked_input_ids.dtype,
        ).view(1, -1).expand((seqlen, -1))

        logits_chunks: list[torch.Tensor] = []
        for start in range(0, seqlen, args.batch_size):
            out = model(
                input_ids=masked_input_ids[start : start + args.batch_size, :],
                attention_mask=expanded_attention_mask[start : start + args.batch_size, :],
                decoder_input_ids=decoder_input_ids[start : start + args.batch_size, :],
                use_cache=False,
            )
            logits_chunks.append(out.logits[:, -1, :].float().detach().cpu())

        logits = torch.cat(logits_chunks, dim=0)  # [L, vocab]
        return torch.log_softmax(logits, dim=-1)  # [L, vocab]

    if args.input_csv is None:
        csv_paths = sorted(p for p in data_dir.glob("*.csv") if p.is_file())
    else:
        p = Path(args.input_csv)
        if not p.is_absolute():
            p = (data_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        csv_paths = [p]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    summaries: list[dict] = []

    for csv_path in csv_paths:
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        missing = sorted(REQUIRED_COLS - set(df.columns))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

        muts_list: list[list[tuple[str, int, str]]] = []
        wt_seqs: list[str] = []

        for row in df.itertuples(index=False):
            mut_info = str(getattr(row, "mut_info")).strip()
            parts = [p.strip() for p in re.split(r"[:;,]+", mut_info) if p.strip()]
            if not parts:
                raise ValueError(f"Empty mut_info: {mut_info!r}")

            muts: list[tuple[str, int, str]] = []
            for part in parts:
                m = _MUT_RE.match(part)
                if not m:
                    raise ValueError(f"Unsupported mut_info format: {mut_info!r}")
                muts.append((m.group(1), int(m.group(2)), m.group(3)))

            mut_seq = str(getattr(row, "seq"))
            seq_chars = list(mut_seq)
            for wt, pos1, mut in muts:
                if pos1 < 1 or pos1 > len(seq_chars):
                    raise ValueError(f"Mutation position out of range: {wt}{pos1}{mut} (len={len(seq_chars)})")
                if seq_chars[pos1 - 1] != mut:
                    raise ValueError(
                        f"Mut AA mismatch for {wt}{pos1}{mut}: seq has {seq_chars[pos1 - 1]}"
                    )
                seq_chars[pos1 - 1] = wt

            muts_list.append(muts)
            wt_seqs.append("".join(seq_chars))

        unique_wt = sorted(set(wt_seqs))
        wt_logps_by_seq: dict[str, torch.Tensor] = {}
        if len(unique_wt) == 1:
            wt_logps_by_seq[unique_wt[0]] = masked_log_probs_all_positions(unique_wt[0])
        else:
            print(f"Found {len(unique_wt)} distinct reconstructed WT sequences; scoring per-WT group.")
            for wt_seq in unique_wt:
                wt_logps_by_seq[wt_seq] = masked_log_probs_all_positions(wt_seq)

        pred: list[float] = []
        for i, (muts, wt_seq) in enumerate(zip(muts_list, wt_seqs, strict=True), start=1):
            logp = wt_logps_by_seq[wt_seq]
            delta = 0.0
            ok = True
            for wt, pos1, mut in muts:
                if wt not in aa_to_id or mut not in aa_to_id:
                    ok = False
                    break
                pos0 = pos1 - 1
                delta += float((logp[pos0, aa_to_id[mut]] - logp[pos0, aa_to_id[wt]]).item())
            pred.append(delta if ok else float("nan"))
            if args.progress_every > 0 and i % args.progress_every == 0:
                print(f"  processed {i}/{len(df)}")

        df["ankh_delta_logp"] = pred

        valid = df["ankh_delta_logp"].notna()
        if valid.any():
            rho, pval = spearmanr(df.loc[valid, "ankh_delta_logp"], df.loc[valid, "fitness"])
        else:
            rho, pval = None, None

        out_csv = output_dir / f"{csv_path.stem}{args.output_suffix}"
        df.to_csv(out_csv, index=False)

        print("\n========== ProteinGym Ankh zero-shot ==========")
        print(f"CSV:          {csv_path.name}")
        print(f"Variants:     {len(df)}")
        if rho is not None:
            print(f"Spearman ρ:   {rho:.4f}")
            print(f"P-value:      {pval:.2e}")
        else:
            print("Spearman ρ:   n/a (no valid predictions)")
        print(f"Saved to:     {out_csv}")
        print("==============================================\n")

        summaries.append(
            {
                "dataset": csv_path.name,
                "variants": int(len(df)),
                "spearman_rho": (float(rho) if rho is not None else None),
                "p_value": (float(pval) if pval is not None else None),
                "output_csv": out_csv.name,
            }
        )

    summary_path = output_dir / "ankh_zeroshot_summary.tsv"
    pd.DataFrame(summaries).to_csv(summary_path, sep="\t", index=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
