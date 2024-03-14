#VisDA UDA
CUDA_VISIBLE_DEVICES=3 python3 UDPCS_MDD.py /DATA/rishabh/UDA/visda -d VisDA2017 -s Synthetic -t Real -a resnet101 --epochs 30 --bottleneck-dim 1024 --seed 0 --train-resizing cen.crop --per-class-eval -b 36 --log logs/VisDA2017
