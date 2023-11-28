.PHONY: 0 1 2 3 4 5 6 7 8 9

all: 0 1 2 3 4 5 6 7 8 9

0: 
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json analyze

1: 0
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json chip --split-ind 0 --num-splits 3

2: 0
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json chip --split-ind 1 --num-splits 3

3: 0
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json chip --split-ind 2 --num-splits 3

4: 1 2 3
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json train

5: 4
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json predict --split-ind 0 --num-splits 3

6: 4
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json predict --split-ind 1 --num-splits 3

7: 4
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json predict --split-ind 2 --num-splits 3

8: 5 6 7
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json eval

9: 8
	python -m rastervision.pipeline.cli run_command /opt/data/output/pipeline-config.json bundle

