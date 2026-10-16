"""Commands: demo, train, recommend. All outputs are local review artifacts."""
import argparse
from pathlib import Path

PROFILES = {"base": 48, "medium": 96}


def train_options(p):
    p.add_argument("--history", type=int, default=24, help="Past hourly context; sparse lags up to this value")
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--profile", choices=list(PROFILES), default="base", help="base: width 48, medium: width 96; training budget stays identical")
    p.add_argument("--hidden", type=int, default=None, help="Explicit width overrides --profile")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=.002)
    p.add_argument("--trees", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--graph", choices=["hybrid", "distance", "identity"], default="hybrid")
    p.add_argument("--neighbors", type=int, default=5)
    p.add_argument("--max-training-rows", type=int, default=250000, help="Evenly sample forecast origins to bound tabular training RAM")
    p.add_argument("--max-eval-origins", type=int, default=96, help="Per partition; selected timestamps are recorded")


def scenario_options(p):
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--ode-steps", type=int, default=8)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")


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
        args.hidden = PROFILES[args.profile] if args.hidden is None else args.hidden
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


if __name__ == "__main__":
    main()
