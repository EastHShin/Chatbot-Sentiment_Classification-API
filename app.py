from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
from flask import Flask, request, jsonify, render_template
from queue import Queue, Empty
from threading import Thread
import torch
import time

app = Flask(__name__)
Chatbot_tokenizer = AutoTokenizer.from_pretrained("EasthShin/Chatbot-LisaSimpson-DialoGPT")
Chatbot_model = AutoModelForCausalLM.from_pretrained("EasthShin/Chatbot-LisaSimpson-DialoGPT")

model_path = "EasthShin/Emotion-Classification-bert-base"
Emotion_tokenizer = AutoTokenizer.from_pretrained(model_path)
Emotion_model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline('text-classification', model=model_path, tokenizer=Emotion_tokenizer)


requests_queue = Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1
init_chat_history_ids = torch.empty(1,1)
step = 0
res = dict()
print("complete model loading")


def handle_requests_by_batch():
    global step
    global res
    while True:
        request_batch = []
        while not (len(request_batch) >= BATCH_SIZE):
            try:
                request_batch.append(requests_queue.get(timeout=CHECK_INTERVAL))

            except Empty:
                continue
            for requests in request_batch:
                try:

                    if step == 0 or step % 3 == 0:
                        step = 0
                        res = make_answer(requests["inputs"][0], init_chat_history_ids, step)
                        requests["output"] = res
                    else:
                        res = make_answer(requests["inputs"][0], res[2], step)
                        requests["output"] = res
                    step += 1

                except Exception as e:
                    requests["output"] = e


handler = Thread(target=handle_requests_by_batch).start()


def make_answer(question, chat_history_ids, step):
    try:

        new_user_input_ids = Chatbot_tokenizer.encode(question + Chatbot_tokenizer.eos_token,
                                                      return_tensors='pt')
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        chat_history_ids = Chatbot_model.generate(bot_input_ids, max_length=1000,
                                                  pad_token_id=Chatbot_tokenizer.eos_token_id)
        res = Chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        result = dict()
        result[0] = classifier(res)[0]
        result[1] = res
        result[2] = chat_history_ids
        return result

    except Exception as e:
        print('Error occur in getting label!', e)
        return jsonify({'error': e}), 500


@app.route('/chat', methods=['POST'])
def chat():

    if requests_queue.qsize() > BATCH_SIZE:
        return jsonify({'message' : 'Invalid request'}), 500

    try:
        args = []
        question = request.form['question']
        args.append(question)
    except Exception as e:
        return jsonify({'message' : 'Invalid request'})
    req = {'inputs': args}
    print(req)
    requests_queue.put(req)
    while 'output' not in req:
        time.sleep(CHECK_INTERVAL)
    print(req['output'])
    result = dict()
    result[0] = req['output'][0]
    result[1] = req['output'][1]
    return result

@app.route('/queue_clear')
def queue_clear():
    while not requests_queue.empty():
        requests_queue.get()

    return 'clear', 200


@app.route('/healthz', methods=['GET'])
def health_check():
    return "Health OK", 200

@app.route('/')
def main():
    return render_template('index.html'), 200

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0')
