.PHONY: help setup draft serve clean stop
.DEFAULT_GOAL = help

help:
	@printf "Usage:\n"
	@grep -E '^[a-zA-Z_-]+:.*?# .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[1;34mmake %-10s\033[0m%s\n", $$1, $$2}'

setup:  # Set up environment for blogging and development.
	bundle install

draft:  # Start a draft blog post.
	( \
	DATE="$(shell date +%Y-%m-%d)"; \
	touch $$DATE-draft.md; \
	echo "---" >> $$DATE-draft.md; \
	echo "title:" >> $$DATE-draft.md; \
	echo "excerpt:" >> $$DATE-draft.md; \
	echo "tags:" >> $$DATE-draft.md; \
	echo "  - " >> $$DATE-draft.md; \
	echo "header:" >> $$DATE-draft.md; \
	echo "  overlay_image: /assets/images/cool-backgrounds/cool-background1.png" >> $$DATE-draft.md; \
	echo "  caption: 'Photo credit: [coolbackgrounds.io](https://coolbackgrounds.io/)'" >> $$DATE-draft.md; \
	echo "last_modified_at:" >> $$DATE-draft.md; \
	echo "search: false" >> $$DATE-draft.md; \
	echo "---" >> $$DATE-draft.md; \
	echo "" >> $$DATE-draft.md; \
	echo "{% if page.noindex == true %}" >> $$DATE-draft.md; \
	echo "  <meta name=\"robots\" content=\"noindex\">" >> $$DATE-draft.md; \
	echo "{% endif %}" >> $$DATE-draft.md; \
	echo "" >> $$DATE-draft.md; \
	mv $$DATE-draft.md _drafts/; \
	)

serve:  # Serve site locally.
	bundle exec jekyll serve --incremental

test:  # Test generated HTML files.
	# Ignore broken links from /r/TheRedPill and Tweets (some people delete tweets)
	bundle exec jekyll build --future
	link_check _site/

compress:  # Compress images losslessly
	jpegoptim assets/images/*.jpg
	optipng assets/images/*.png
	optipng assets/images/cool-backgrounds/*.png

clean:  # Remove _site/ and _posts/_site/ directories
	rm -rf Gemfile.lock package-lock.json _site/ _posts/_site/ vendor/
	jekyll clean
