"""Conversational AI (NLU) pipeline:

1. Bag-of-Words logistic regression baseline (intent only)
2. JointBERT-mini trained from scratch (intent + slots)
3. JointBERT with a fine-tuned pretrained DistilBERT backbone (teacher)
4. MLM-augmented knowledge distillation teacher -> JointBERT-mini (deployed model)

    python scripts/train_nlu.py
    python scripts/train_nlu.py --skip-bert      # steps 1-2 only, no download needed
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgebot.nlu.data import INTENTS  # noqa: E402
from edgebot.nlu.engine import train_bow_baseline, train_nlu  # noqa: E402
from edgebot.utils import CHECKPOINTS, RESULTS, ensure_dirs, get_device, save_json  # noqa: E402

DEMO_SENTENCES = [
    "hey jetbot move the red box from shelf a to the conveyor",
    "uh could you grab the blew crate on shelf b",
    "stack the green blue and red blocks at the workbench",
    "head over to the charging dock please",
    "what are you carrying right now",
    "haul the yellow parcel over to the packing station",
]


def summarise(name, m):
    print(f"{name:<28s} intent {m['intent_acc']:.4f} | slot-F1 {m['slot_f1']:.4f} | frame {m['frame_acc']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-bert", action="store_true")
    p.add_argument("--bert-epochs", type=int, default=4)
    p.add_argument("--aug-variants", type=int, default=4)
    args = p.parse_args()
    ensure_dirs()
    device = get_device(args.device)

    nlu, splits, history = train_nlu(args.epochs, args.seed, device)
    results = {"intents": INTENTS, "sizes": {k: len(v) for k, v in splits.items()}, "models": {}}
    results["models"]["bow_logreg"] = {"intent_acc": train_bow_baseline(splits)}

    def record(name, model_nlu, extra=None):
        entry = {"n_params": sum(p.numel() for p in model_nlu.model.parameters()), "vocab_size": len(model_nlu.vocab)}
        for split in ("val", "test", "test_noisy"):
            entry[split] = model_nlu.evaluate(splits[split])
            summarise(f"{name} [{split}]", entry[split])
        entry.update(extra or {})
        results["models"][name] = entry

    record("jointbert_mini_scratch", nlu)
    save_json(history, RESULTS / "nlu_history.json")
    deployed = nlu

    if not args.skip_bert:
        from edgebot.nlu.bert import train_joint_bert
        from edgebot.nlu.distill import distill, mlm_augment

        teacher, bert_res = train_joint_bert(splits, epochs=args.bert_epochs, device=device, seed=args.seed)
        for split in ("test", "test_noisy"):
            summarise(f"jointbert_distilbert [{split}]", bert_res[split])
        results["models"]["jointbert_distilbert"] = bert_res
        augmented = mlm_augment(splits["train"], n_variants=args.aug_variants, device=device, seed=args.seed)
        print(f"MLM augmentation produced {len(augmented)} new utterances, e.g.:")
        for e in augmented[:8]:
            print("   ", e.text)
        results["augmentation"] = {"n_augmented": len(augmented), "samples": [e.text for e in augmented[:25]]}
        student = distill(teacher, splits, augmented, device=device, epochs=args.epochs, seed=args.seed)
        record("jointbert_mini_distilled", student)
        deployed = student
        del teacher

    print("BoW baseline", results["models"]["bow_logreg"])
    deployed.model.to("cpu")
    deployed.device = deployed.device.__class__("cpu")
    results["deployed"] = "jointbert_mini_distilled" if deployed is not nlu else "jointbert_mini_scratch"
    results["examples"] = [{"text": s, **deployed.parse(s).to_dict()} for s in DEMO_SENTENCES]
    for ex in results["examples"]:
        print(ex)
    deployed.save(str(CHECKPOINTS / "nlu_jointbert_mini.pt"), {"seed": args.seed, "variant": results["deployed"]})
    save_json(results, RESULTS / "nlu_eval.json")


if __name__ == "__main__":
    main()
