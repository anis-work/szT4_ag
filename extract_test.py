from pypdf import PdfReader
r = PdfReader('resumes/alice_johnson.pdf')
print('pages:', len(r.pages))
text = r.pages[0].extract_text()
print('text len:', len(text) if text else 0)
print('text:', repr(text[:300]) if text else 'EMPTY')
