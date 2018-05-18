build:
	docker build -t sli --rm .

clean:
	docker rmi sli $(shell docker images -f 'dangling=true' -q)

.PHONY: build
