Quick note on how to compile papers:

Compile your paper (from repo root)
```latexmk -pdf neurips_2024.tex```
Open the PDF
```open neurips_2024.pdf```
If you change the .tex, just re-run:
```latexmk -pdf neurips_2024.tex```

Bibliography/citations:
- References live in `references.bib` and are included from `neurips_2024.tex` via BibTeX.
- If you compile without `latexmk`, run: `pdflatex neurips_2024.tex && bibtex neurips_2024 && pdflatex neurips_2024.tex && pdflatex neurips_2024.tex`.
