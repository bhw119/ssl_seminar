from __future__ import annotations

import json
import pathlib
import tempfile

from sslimg.smoke import fake_dataset

from .run import ABLATIONS, METHODS, ReproCfg, run_grid, run_path, tables

def main():
    ok = []

    def check(name, cond):
        ok.append(bool(cond))
        print(("  ok  " if cond else "  FAIL ") + name)

    cfg = ReproCfg(iters=6, bs=4, uratio=1, img_size=32, eval_every=3, num_classes=6, n_ulb=150, pretrained=False,
                   workers=0, cg_warmup=2, cg_queue=2)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        root, out = tmp / "raw", tmp / "out"
        fake_dataset(root)
        methods = METHODS + ABLATIONS
        run_grid(str(root), str(out), methods, ("in",), (0,), cfg)
        runs = {m: json.loads(run_path(str(out), "in", m, 0).read_text()) for m in methods if run_path(str(out), "in", m, 0).exists()}
        check(f"실행 {len(methods)}개 모두 결과 JSON 저장", len(runs) == len(methods))
        check("test 레이블은 solution.csv에서, 정확도는 0~1", all(0 <= r["test"]["acc"] <= 1 for r in runs.values()))
        check("test 36장, 종 6개", all(len(r["test"]["per_class_f1"]) == 6 for r in runs.values()))
        check("학습 곡선 기록", all(len(r["history"]) >= 2 for r in runs.values()))
        check("SC-SSL: head 4종(h1, h2, 보정 전후) 기록", set(runs["scssl"]["extra"]["heads_test_acc"]) == {"h1", "h2", "h1_raw", "h2_raw"})
        check("InterLUDE: 평가 때 fusion 3종 기록", set(runs["interlude"]["extra"]["ef_at_eval_test_acc"]) == {"no_fusion", "fusion_file_order", "fusion_shuffled"})
        check("마지막 스텝 체크포인트도 평가", all(r["extra"]["last"]["it"] == cfg.iters for r in runs.values()))
        t = tables(str(out))
        check("표: test_acc, gain_vs_fixmatch, ablation, SC-SSL head, InterLUDE fusion",
              {"test_acc", "gain_vs_fixmatch", "ablation_test_acc", "ablation_gain_vs_full", "scssl_heads", "interlude_ef_at_eval"} <= set(t))
        run_grid(str(root), str(out), methods, ("in",), (0,), cfg)
        check("끝난 실행은 건너뜀", True)
    print(f"\n{sum(ok)}/{len(ok)} 통과")
    raise SystemExit(0 if all(ok) else 1)

if __name__ == "__main__":
    main()
