"""Commands: demo, train, recommend. All outputs are local review artifacts."""
import argparse
from pathlib import Path

from .profiles import PROFILES, resolve_profile


def train_options(p):
    p.add_argument("--history", type=int, default=24, help="Past hourly context; sparse lags up to this value")
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--profile", choices=list(PROFILES), default="base", help="base 48; medium 96; large 256; rtx3080 512; 100m 1024 (99.96M parameters)")
    p.add_argument("--hidden", type=int, default=None, help="Explicit width overrides --profile")
    p.add_argument("--gcn-layers", type=int, default=None)
    p.add_argument("--flow-blocks", type=int, default=None)
    p.add_argument("--expansion", type=int, default=None, help="Feed-forward expansion in additional residual flow blocks")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Training microbatch size")
    p.add_argument("--accumulate", type=int, default=None, help="Microbatches per optimizer update")
    p.add_argument("--patience", type=int, default=None, help="Stop after this many unimproved validation epochs; 0 disables")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--trees", type=int, default=None)
    p.add_argument("--xgb-depth", type=int, default=None)
    p.add_argument("--xgb-device", choices=["cpu", "cuda"], default=None, help="Explicit XGBoost training device")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Forecast origins per evaluation batch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph", choices=["hybrid", "distance", "identity"], default="hybrid")
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--max-training-rows", type=int, default=250000, help="Evenly sample forecast origins to bound tabular training RAM")
    p.add_argument("--max-eval-origins", type=int, default=96, help="Per partition; selected timestamps are recorded")


def scenario_options(p):
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--ode-steps", type=int, default=None)
    p.add_argument("--sample-chunk", type=int, default=None, help="Flow scenarios generated together; limits GPU activation memory")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA float16 mixed precision; CPU remains float32")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)


def plan_options(p):
    p.add_argument("--budget", type=int, default=20, help="Maximum total number of bikes transferred")
    p.add_argument("--truck-capacity", type=int, default=10, help="Maximum bikes per direct-transfer batch")
    p.add_argument("--service-level", type=float, default=.9, help="Scenario quantile for desired opening stock")
    p.add_argument("--max-distance-km", type=float, default=5.)


def main(argv=None):
    p = argparse.ArgumentParser(description="타슈 공모전: 세 모델 비교, 품절 예측, 이동 대수 추천")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run the entire pipeline using clearly labeled synthetic data")
    demo.add_argument("--out", required=True)
    demo.add_argument("--days", type=int, default=35)
    demo.add_argument("--stations", type=int, default=12)
    train_options(demo)
    scenario_options(demo)
    plan_options(demo)
    train = sub.add_parser("train", help="Fit all three models on a measured hourly CSV panel")
    train.add_argument("--panel", required=True)
    train.add_argument("--train-end", required=True, help="Exclusive KST training end, e.g. 2026-02-01")
    train.add_argument("--valid-end", required=True, help="Exclusive KST validation end")
    train.add_argument("--out", required=True)
    train_options(train)
    scenario_options(train)
    rec = sub.add_parser("recommend", help="Load saved models and propose concrete direct bike transfers")
    rec.add_argument("--run", required=True)
    rec.add_argument("--panel", required=True)
    rec.add_argument("--at", help="Forecast origin hour in KST; defaults to last panel row")
    rec.add_argument("--snapshot", help="Optional measured station_id,stock CSV at --at")
    rec.add_argument("--model", choices=["auto", "xgboost", "gcn", "gcn_flow"], default="auto")
    rec.add_argument("--out", required=True)
    scenario_options(rec)
    plan_options(rec)
    args = p.parse_args(argv)
    if args.command in ("train", "demo"):
        resolve_profile(args)
    try:
        from .pipeline import fit, recommend
        if args.command == "train":
            args.source = "user_provided_unverified"
            fit(args)
        elif args.command == "recommend":
            recommend(args)
        else:
            from .data import make_demo
            out = Path(args.out)
            if out.exists() and any(out.iterdir()):
                raise ValueError("Choose an empty demo output directory.")
            args.panel = out / "synthetic_panel.csv"
            args.train_end, args.valid_end = map(str, make_demo(args.panel, args.days, args.stations, args.seed))
            args.source = "synthetic"
            args.out = str(out / "run")
            fit(args)
            args.run, args.at, args.snapshot, args.model = args.out, None, None, "auto"
            args.out = str(out / "recommendation")
            recommend(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        p.exit(2, f"입력/실행 오류: {exc}\n")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            p.exit(2, "GPU/메모리 부족: train/demo는 --batch-size 2 --accumulate 8 --eval-batch-size 1 --sample-chunk 1, recommend는 --sample-chunk 1을 사용하고 새 출력 폴더로 재실행하세요.\n")
        raise


if __name__ == "__main__":
    main()
