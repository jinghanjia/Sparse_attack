Train BigConv:
```bash
python train.py
```

Train MoEBigConv:
```bash
python train.py ----model-type BigConvMoE
```
or with pretrained BigConv model
```bash
python train.py --model-type BigConvMoE --p pretrained/path --lr-max 0.01
```