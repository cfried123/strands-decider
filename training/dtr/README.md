# DTR Part I fine-tune

Domain fine-tune of the released v21 checkpoint on DTR Part I (Passenger Movement)
determinations. Source chapters: https://www.business.ustranscom.mil/dtr/dtrp1.cfm.

## Data

```bash
# 10 task families; Ch. 106 (rental) and Ch. 107 (ship) are held out of training.
strands-decider data build \
  --recipe dtr_class_of_service --recipe dtr_travel_mode --recipe dtr_compliance_risk \
  --recipe dtr_fare_eligibility --recipe dtr_cpp_exception --recipe dtr_foreign_carrier \
  --recipe dtr_contractor_airlift --recipe dtr_booking_policy --recipe dtr_provision_routing \
  --recipe dtr_rental_use --recipe dtr_ship_travel \
  --holdout dtr_rental_use --holdout dtr_ship_travel --out data/dtr/train.jsonl

# DTR-Bench (142 items, disjoint scenario vocabulary; self-reviewed, SME review pending)
python -m strands_decider.data.dtr --out data/dtr_bench.jsonl
```

Ambiguous yes/no scenarios (the decisive fact is missing) are written as two rows,
labels 0 and 1 at weight 0.5, i.e. a soft 0.5 target; the model learns to say "I don't
know" there, which is what routes a case to human review.

## Train, evaluate, export

```bash
strands-decider train -c training/dtr/finetune_dtr.yaml   # LoRA, head_lr > lr, shuffle, EMA
python -m strands_decider.dtr_eval checkpoints/dtr-ft --out results/dtr/bench_ft.json
python -m strands_decider.hf_export checkpoints/dtr-ft <export dir>
```

`finetune_from` continues training the released checkpoint's adapter *and* head
(unlike `init_from`, which replaces the head and freezes the torso).

Review policy used for scoring (and by the app): a yes/no abstains when
|P(true) - 0.5| < 0.25; a choice or score abstains when its top probability is < 0.5.
