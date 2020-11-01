#! /bin/bash

rm plots/*
python plot.py
pandoc docs/index.md -s -V mainfont="TeX"  -o pages/index.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --filter pandoc-plot --katex
pandoc docs/reinforcement.md -s -V mainfont="TeX"  -o pages/reinforcement.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --filter pandoc-plot --katex
pandoc docs/unsupervised.md -s -V mainfont="TeX"  -o pages/unsupervised.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --filter pandoc-plot --katex
