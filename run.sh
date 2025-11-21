#!/bin/bash
# git pull --quiet
python -m manga_translator local --use-gpu -v -i 'G:\manga\A' --config-file 'E:\MangaTranslator\manga-image-translator\examples\config-example.toml'
python -m manga_translator local --use-gpu -v -i 'G:\manga\E' --config-file 'E:\MangaTranslator\manga-image-translator\examples\config-example.toml'
