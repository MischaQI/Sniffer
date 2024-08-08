GPU=$1
PORT=$2
python -m torch.distributed.run --nproc_per_node=${GPU} --master_port=${PORT} evaluate.py --cfg-path lavis/projects/instructblip2/eval/factvqa_newsclip_eval.yaml ${@:3}


