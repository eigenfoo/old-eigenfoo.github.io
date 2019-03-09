.PHONY: help serve kill
.DEFAULT_GOAL = help

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

serve:  # Serve site locally.
	nohup jekyll serve --incremental &

kill:  # Kill jekyll process.
	pkill jekyll
	rm nohup.out

clean:  # Remove _site/ directories.
	rm -rf nohup.out _site/ _posts/_site/
