# Minimal makefile for Sphinx documentation

SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Default target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile clean

# Standard sphinx-build targets
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Clean target: remove the entire build directory
clean:
	@echo "Removing build directory..."
	@rm -rf "$(BUILDDIR)"
	@echo "Done."

publish:
	@echo "Building HTML..."
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Deploying to GitHub Pages..."
	ghp-import -n -p -r github $(BUILDDIR)/html
	@echo "Done. GitHub Pages updated."

