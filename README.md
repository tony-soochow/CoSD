# CoSD: Balancing Behavioral Consistency and Diversity in Unsupervised Skill Discovery


## Installation
```bash
pip3 install -r requirements.txt
```

## Train Skills
```shell
python3 main.py --mem_size=1000000 --env_name="Hopper-v3" --interval=100 --do_train --n_skills=10
```
## Employing Skills in Hierarchical RL
```shell
python run_task.py
```