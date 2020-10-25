#! /bin/bash

rm plots/*
pandoc docs/index.md -s --filter pandoc-plot --katex -o index.html -c style.css --toc
