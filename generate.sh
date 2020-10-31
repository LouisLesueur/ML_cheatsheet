#! /bin/bash

rm plots/*
python plot.py
pandoc docs/index.md -s -V mainfont="TeX"  -o index.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --filter pandoc-plot --katex
