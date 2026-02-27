# Makefile for HIRT Project Documentation
# Generates PDFs from Markdown using pandoc

# Configuration
PANDOC = pandoc
PANDOC_OPTS = --pdf-engine=xelatex \
              --variable=geometry:margin=1in \
              --variable=fontsize:11pt \
              --variable=documentclass:article \
              --toc \
              --toc-depth=3 \
              --number-sections

# Directories
WHITEPAPER_DIR = docs/whitepaper
SECTIONS_DIR = $(WHITEPAPER_DIR)/sections
PDF_DIR = $(WHITEPAPER_DIR)/pdf
FIELD_GUIDE_DIR = docs/field-guide

# Source files
MAIN_MD = $(WHITEPAPER_DIR)/main.md
SECTIONS = $(wildcard $(SECTIONS_DIR)/*.md)
FIELD_GUIDE_MD = $(wildcard $(FIELD_GUIDE_DIR)/*.md)

# Output files
MAIN_PDF = $(PDF_DIR)/HIRT_Whitepaper.pdf
FIELD_GUIDE_PDF = $(PDF_DIR)/HIRT_FieldGuide.pdf

# Default target
all: whitepaper field-guide

# Create PDF directory if it doesn't exist
$(PDF_DIR):
	mkdir -p $(PDF_DIR)

# Whitepaper PDF (combines main + all sections)
whitepaper: $(PDF_DIR) $(MAIN_PDF)

$(MAIN_PDF): $(MAIN_MD) $(SECTIONS)
	$(PANDOC) $(PANDOC_OPTS) \
		--metadata title="DIY Probe-Array Subsurface Imaging System (MIT-3D + ERT-Lite)" \
		--metadata author="HIRT Project" \
		--metadata date="$(shell date +%Y-%m-%d)" \
		-o $@ \
		$(MAIN_MD) $(SECTIONS)

# Field Guide PDF
field-guide: $(PDF_DIR) $(FIELD_GUIDE_PDF)

$(FIELD_GUIDE_PDF): $(FIELD_GUIDE_MD)
	$(PANDOC) $(PANDOC_OPTS) \
		--metadata title="HIRT Field Guide" \
		--metadata author="HIRT Project" \
		--metadata date="$(shell date +%Y-%m-%d)" \
		-o $@ \
		$(FIELD_GUIDE_MD)

# Individual section PDFs (optional)
sections: $(PDF_DIR)
	@for section in $(SECTIONS); do \
		output=$$(basename $$section .md); \
		$(PANDOC) $(PANDOC_OPTS) -o $(PDF_DIR)/$$output.pdf $$section; \
	done

# Clean generated PDFs
clean:
	rm -f $(PDF_DIR)/*.pdf

# Clean all (including directory)
clean-all: clean
	rmdir $(PDF_DIR) 2>/dev/null || true

# Help target
help:
	@echo "HIRT Documentation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Generate all PDFs (default)"
	@echo "  whitepaper   - Generate main whitepaper PDF"
	@echo "  field-guide  - Generate field guide PDF"
	@echo "  sections     - Generate individual section PDFs"
	@echo "  clean        - Remove generated PDFs"
	@echo "  clean-all    - Remove PDFs and PDF directory"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - pandoc (https://pandoc.org/)"
	@echo "  - XeLaTeX (for PDF generation)"
	@echo ""
	@echo "Example:"
	@echo "  make whitepaper    # Generate whitepaper PDF"
	@echo "  make all           # Generate all PDFs"

.PHONY: all whitepaper field-guide sections clean clean-all help

