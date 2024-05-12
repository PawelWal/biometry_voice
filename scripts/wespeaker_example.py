import wespeaker

audio_path = 'test.wav'
audio_path2 = 'gentle-emerald-parrotfish_1.wav'

model = wespeaker.load_model('english')
model.set_gpu(1)
embedding = model.extract_embedding(audio_path)
print("EMBEDDING ", embedding.shape, embedding[:10])
# utt_names, embeddings = model.extract_embedding_list('wav.scp')
similarity = model.compute_similarity(audio_path, audio_path2)
print("SIMILARITY ", similarity)

similarity = model.compute_similarity(audio_path, audio_path)
print("SIMILARITY ", similarity)

# diar_result = model.diarize(audio_path2)
# print("DIARIZATION ", diar_result)

# register and recognize
model.register('spk1', audio_path)
model.register('spk2', audio_path2)
result = model.recognize(audio_path)
print("RESULT for first user", result)
result = model.recognize(audio_path2)
print("RESULT for second user", result)
