"""Model capacity and compute defaults, shared by CLI and metadata validation."""

BASE = dict(hidden=48, gcn_layers=2, flow_blocks=1, expansion=2, dropout=0.,
            epochs=15, batch_size=32, accumulate=1, lr=.002, patience=0,
            trees=150, xgb_depth=4, xgb_device="cpu", device="cpu",
            samples=32, ode_steps=8, sample_chunk=32, eval_batch_size=8)
PROFILES = {
    "base": dict(BASE),
    "medium": dict(BASE, hidden=96),
    "large": dict(BASE, hidden=256, gcn_layers=4, flow_blocks=4, dropout=.1,
                  epochs=40, batch_size=8, accumulate=2, lr=.001, patience=10,
                  trees=300, xgb_depth=6, device="cuda", xgb_device="cuda",
                  samples=64, ode_steps=12, sample_chunk=4, eval_batch_size=2),
    "rtx3080": dict(BASE, hidden=512, gcn_layers=4, flow_blocks=6, dropout=.1,
                    epochs=60, batch_size=4, accumulate=4, lr=.0005, patience=12,
                    trees=600, xgb_depth=6, device="cuda", xgb_device="cuda",
                    samples=64, ode_steps=12, sample_chunk=4, eval_batch_size=2),
}


def resolve_profile(args):
    for key, value in PROFILES[args.profile].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    return args


def architecture_label(args):
    keys = ("hidden", "gcn_layers", "flow_blocks", "expansion", "dropout")
    return args.profile if all(getattr(args, k) == PROFILES[args.profile][k] for k in keys) else "custom"
