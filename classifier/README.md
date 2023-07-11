# Data
## Train

| Relation | Source | Mutable? | Template |  Count | Comment |
|---|---|---|---|---|---|
| P286 | TempLAMA | 1 | [X]'s head coach is [Y] | 796 | Rewritten to avoid all the mutable templates finishing in "of" |
| P6 | TempLAMA | 1 | The head of government of [X] is [Y] | 428 | See P286's comment |
| P102 | TempLAMA | 1 | [X] is a member of [Y] | 731 | Rewritten to have [Y] at the end |
| P937 | LAMA | 1 | [X] found employment in [Y]<br/> [X] took up work in [Y]<br/> [X] used to work in [Y]<br/> [X] was employed in [Y]<br/> [X] worked in [Y]<br/> | 402 |  |
| | | 1 | | 2357 | |
| P19 | LAMA | 0 | [X] is native to [Y]<br/> [X] is originally from [Y]<br/> [X] originated from [Y]<br/> [X] originates from [Y]<br/> [X] was born in [Y]<br/> [X] was native to [Y]<br/> [X] was originally from [Y]<br/> | 390 |  |
| P20 | LAMA | 0 | [X] died at [Y]<br/> [X] died in [Y]<br/> [X] expired at [Y]<br/> [X] lost their life at [Y]<br/> [X] passed away at [Y]<br/> [X] passed away in [Y]<br/> [X] succumbed at [Y]<br/> [X]'s life ended in [Y]<br/> | 407 |  |
| P103 | LAMA | 0 | The mother tongue of [X] is [Y]<br/> The native language of [X] is [Y]<br/> | 448 |  |
| P127 | LAMA | 0 | [X] is owned by [Y]<br/> [X] owner [Y]<br/> | 336 |  |
| | | 0 | | 1581 | |

## Validation
| Relation | Source | Mutable? | Template | Count | Comment |
|---|---|---|---|---|---|
| P108 | TLAMA | 1 | [X] works for [Y] | 943 | Rewritten to have [Y] at the end |
| P488 | TLAMA | 1 | The role of chair at [X] is occupy by [Y] | 444 | Rewritten to not match too much the template of P102 in the train |
| P159 | LAMA | 0 | The headquarter of [X] is in [Y]</br> The headquarter of [X] is located in [Y]</br> The headquarters of [X] is in [Y]</br> [X]'s headquarters are in [Y]</br> [X], whose headquarters are in [Y]</br> [X] is based in [Y]</br> [X] is headquartered in [Y] | 791 |  |
| P364 | LAMA | 0 | The language of [X] is [Y]</br> The language of [X] was [Y]</br> The original language of [X] is [Y]</br> The original language of [X] was [Y] | 747 |  |

## Test
| Relation | Source | Mutable? | Template | Count | Comment |
|---|---|---|---|---|---|
| P39 | TLAMA | 1 | [X] serves as [Y] | 1000 | Rewritten to not match too much the template of P102 in the train |
| P54 | TLAMA | 1 | [X] is associated with [Y] | 1000 | Original [X] plays for [Y] |
| P449 | LAMA | 0 | [X] was originally aired on [Y]</br> [X] debuted on [Y]</br> [X] is to debut on [Y]</br> [X] premiered on [Y]</br> [X] premieres on [Y]</br> [X] was released on [Y] | 795 |  |
| P36 | LAMA | 0 | [X], which has the capital [Y]</br> [X]'s capital is [Y]</br> [X]'s capital, [Y]</br> [X]'s capital city, [Y]</br> [X]'s capital city is [Y]</br> [X], which has the capital city [Y]</br> The capital city of [X] is [Y]</br> The capital of [X] is [Y] | 440 |  |
| P413 | LAMA | 0 | [X] plays as [Y]</br> [X] plays in the position of [Y] | 952 | The football position in which a person plays. |

# Results
Run sweep in wandb
```yaml
method: random
metric:
  goal: minimize
  name: eval/loss
parameters:
  evaluation_strategy:
    value: epoch
  learning_rate:
    distribution: uniform
    max: 0.0001
    min: 1e-06
  load_best_model_at_end:
    value: true
  logging_strategy:
    value: epoch
  lr_scheduler_type:
    values:
      - linear
      - cosine
      - constant
  metric_for_best_model:
    value: loss
  model_name_or_path:
    value: /projects/nlp/data/constanzam/llama/huggingface-ckpts/7B
  num_train_epochs:
    values:
      - 5
      - 8
  per_device_train_batch_size:
    values:
      - 8
      - 16
      - 32
  save_strategy:
    value: epoch
  save_total_limit:
    value: 1
  warmup_ratio:
    values:
      - 0
      - 0.1
      - 0.2
  weight_decay:
    value: 0.01
program: run_classifier.py
```
Best hparams:
- learning_rate=6.497177543076478e-06
- num_train_epochs=8
- per_device_train_batch_size=8
- warmup_ratio=0.1

## Evaluation
| Split | Accuracy Mutable | Accuracy Immutable |
|---|---|---|
| Train | 99.7 | 97.4 |
| Validation | 89.9 | 97.9 |
| Test | 85.4 | 48.5 |
| Test (-P413) | 85.4 | 79.8 |

Obs: The accuracy on the P413 relation is 0.07, maybe because it's being representated quite similar to P54? Both are about football players, but one P54 is about the club where the person plays and the other the position.