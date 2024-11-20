# coding: iso-8859-1 -*-


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def removing_timestamps(text):
    output = []
    counter = 1
    for line in text:
        if counter % 3 == 0:
            output.append(line)
        if line != '':
            counter += 1
    return output


def extract_output_diagnosis(llm_output_text):
    sentences = []
    x = llm_output_text.find('<START>', llm_output_text.find('<START>') + 1)
    y = llm_output_text.find('<END>', llm_output_text.find('<END>') + 1)
    sentence = llm_output_text[x+7:y].replace('\n', '')
    sentences.append(sentence)

    while llm_output_text.find('<START>', x + 1) > 0:
        x = llm_output_text.find('<START>', x + 1)
        y = llm_output_text.find('<END>', y + 1)
        sentence = llm_output_text[x+7:y].replace('\n', '')
        sentences.append(sentence)

    return sentences


device = "cuda" # the device to load the model onto

model_name = "speakleash/Bielik-11B-v2.2-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, token="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")

f = open('test/2024-10-09_06-17-46.srt')
#f = open('test/2021-12-14-03-06-09_segmentation_large_dialog.txt')
#f = open('test/2021-12-14-03-06-09_segmentation_large_dialog.txt')
#f = open('test/2024-01-11-16-19-25_segmentation_large_dialog.txt')

text = f.read()
splitted_text = text.split('\n')

output_text = removing_timestamps(splitted_text)
output_text = ' '.join(output_text)


f_ = open('test-17-46.txt', 'w+')

messages = [{"role": "system", "content": "Czy mozesz wskazac czesc dialogu gdzie jest mowa o postawieniu diagnozy przez lekarza? Wypowiedzi oddzielone sa znakiem nowej linii. Dialog to: " + output_text + ". Nie dodawaj zadnego swojego tekstu ani komentarzy. Wynikiem maja byc fragmenty poswiecone tylko postawionej diagnozie lekarskiej. Fragmenty te maja byc umieszczone pomiedzy znacznikami <START> i <END>"}]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = input_ids.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=4000, do_sample=True, top_k=1)
decoded__ = tokenizer.batch_decode(generated_ids)
diagnosis = extract_output_diagnosis(decoded__[0])

messages = [{"role": "system", "content": "Czy mozesz wskazac czesc dialogu gdzie jest mowa o dolegliwosciach? Wypowiedzi oddzielone sa znakiem nowej linii. Dialog to: " + output_text + ". Nie dodawaj zadnego swojego tekstu ani komentarzy. Wynikiem maja byc fragmenty poswiecone tylko dolegliwosciom. Fragmenty te maja byc umieszczone pomiedzy znacznikami <START> i <END>"}]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = input_ids.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=4000, do_sample=True, top_k=1)
decoded__ = tokenizer.batch_decode(generated_ids)

symptoms_messages = extract_output_diagnosis(decoded__[0])

messages = [{"role": "system", "content": "Mamy nastepujacy tekst: " + ' '.join(symptoms_messages) + ". Prosze wyciagnij z niego objawy tak aby kazdy objaw byl w osobnej linii. Nie dodawaj zadnego swojego tekstu ani komentarzy. Wynikiem maja byc fragmenty poswiecone tylko objawom. Fragmenty te maja byc umieszczone pomiedzy znacznikami <START> i <END>"}]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = input_ids.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=4000, do_sample=True, top_k=1)
decoded__ = tokenizer.batch_decode(generated_ids)

symptoms = extract_output_diagnosis(decoded__[0])

#messages = [{"role": "system", "content": "Odpowiedz czy katar jest wodnisty z ponizszego dialogu: Woda sie z nosa leje? No taki gesciejszy jest. A w kazdym razie nie jest to geste, zielone? Jeszcze nie. Nie dodawaj zadnego swojego tekstu ani komentarzy. Wynikiem ma byc odpowiedz TAK jesli jest wodnisty, natomiast NIE jesli nie jest wodnisty. Odpowiedz ma byc umieszczona pomiedzy znacznikami <START> i <END>"}]
#input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

#model_inputs = input_ids.to(device)
#model.to(device)

#generated_ids = model.generate(model_inputs, max_new_tokens=4000, do_sample=True, top_k=1)
#decoded__ = tokenizer.batch_decode(generated_ids)

#katar = extract_output_diagnosis(decoded__[0])

messages = [{"role": "system", "content": "Spróbuj znaleźć i podać jakie leki są wymienione w ponizszym dialogu: " + output_text + "Nie dodawaj zadnego swojego tekstu ani komentarzy. Wynikiem ma byc lista leków z dialogu lekarza z pacjentem. Odpowiedz ma byc umieszczona pomiedzy znacznikami <START> i <END>"}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = input_ids.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=4000, do_sample=True, top_k=1)
decoded__ = tokenizer.batch_decode(generated_ids)

drugs = extract_output_diagnosis(decoded__[0])

f_.write('Diagnoza:\n')
f_.write(' '.join(diagnosis))
f_.write('Rozpoznanie i objawy:\n')
f_.write(' '.join(symptoms))
f_.write('Leki:\n')
f_.write(' '.join(drugs))
f_.close()

