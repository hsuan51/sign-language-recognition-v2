from google.cloud import translate
client = translate.TranslationServiceClient()

project_id = '805170059749'
text = 'you time_a have'
location = 'us-central1'
model = 'projects/805170059749/locations/us-central1/models/TRL7393067806654201856'

parent = client.location_path(project_id, location)

response = client.translate_text(
    parent=parent,
    contents=[text],
    model=model,
    mime_type='text/plain',  # mime types: text/plain, text/html
    source_language_code='en',
    target_language_code='zh')

for translation in response.translations:
  print('Translated Text: {}'.format(unicode(translation).encode('utf8')))
