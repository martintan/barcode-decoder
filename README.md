### convert pdf to jpg

```bash
brew install poppler

python pdf_to_jpg.py "/Users/martintan/Downloads/12-2-2019_unrecognized/[Barcode]-20191202-102119-00500002.pdf"
```

### install torch with cuda (windows)

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```