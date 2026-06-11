# Slide building

To build the slides with `pandoc`:

    pandoc session2_slides_pandoc.md -t beamer --slide-level=2 -V theme=metropolis -V fontsize=10pt -o session2/session2_slides.pdf
