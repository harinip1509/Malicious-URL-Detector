CSV ?= data/malicious_phish.csv
ROWS ?= 60000

install:
	pip install -r requirements.txt
	cd frontend && npm install

eda:
	python3 -m src.eda --csv $(CSV)

train:
	python3 -m src.train --csv $(CSV) --max-rows $(ROWS)

api:
	python3 -m backend.app

ui:
	cd frontend && npm run dev

clean:
	rm -rf artifacts/* reports/*.csv reports/*.png
	touch artifacts/.gitkeep reports/.gitkeep
