#! /bin/bash

rm docs/plots/*
python plot.py
pandoc md/index.md -s -V mainfont="TeX"  -o docs/index.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --katex
pandoc md/reinforcement.md -s -V mainfont="TeX"  -o docs/reinforcement.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --katex
pandoc md/unsupervised.md -s -V mainfont="TeX"  -o docs/unsupervised.html -c style.css --toc --template pandoc-bootstrap/template.html --include-in-header pandoc-bootstrap/header.html --include-before-body pandoc-bootstrap/navbar.html --include-after-body pandoc-bootstrap/footer.html --katex
