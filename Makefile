PY ?= python

.PHONY: install test tabular nav manipulation nlu edge voice demo figures report all

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

tabular:
	$(PY) scripts/run_tabular.py --runs 20

nav:
	$(PY) scripts/train_nav_ppo.py --steps 5000000
	$(PY) scripts/train_nav_ppo.py --steps 3000000 --action discrete
	$(PY) scripts/train_nav_dqn.py --obs lidar --steps 1000000
	$(PY) scripts/train_nav_dqn.py --obs pixels --steps 1000000

manipulation:
	$(PY) scripts/run_manipulation.py --trials 100
	$(PY) scripts/train_arm_reach.py --steps 2000000

nlu:
	$(PY) scripts/train_nlu.py

edge:
	$(PY) scripts/export_edge.py

voice:
	$(PY) scripts/eval_voice.py --n 150

demo:
	$(PY) scripts/run_demo.py --voice --video results/media/edgebot_demo.mp4

figures:
	$(PY) scripts/make_figures.py

report:
	cd docs && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex
	cd docs && pdflatex -interaction=nonstopmode slides.tex && pdflatex -interaction=nonstopmode slides.tex

all: tabular nav manipulation nlu edge voice demo figures report
