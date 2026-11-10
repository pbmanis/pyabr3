# Build the API documentation, using the "docs" dependency group in pyproject.toml.
#     make -C doc html

SPHINXBUILD = uv run --group docs sphinx-build

.PHONY: html

html:
	cd .. && $(SPHINXBUILD) -b html doc doc/build/html
