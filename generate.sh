#! /bin/bash

pandoc docs/index.md -f markdown -t html -s --katex -o index.html -c style.css
