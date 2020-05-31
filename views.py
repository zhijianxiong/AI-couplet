
from django.shortcuts import render,HttpResponse
# Create your views here.

from django.contrib import messages
import linecache
import random
import sys
sys.path.append(r'F:\001')
from model import Model


def input_form(request):
    return render(request,'1.htm')




def input(request):
    vocab_file = 'F:/001/data/dl-data/couplet/vocabs'
    model_dir = 'F:/001/data/dl-data/models/tf-lib/output_couplet'
    request.encoding = 'utf-8'
    up = ''
    output = ''
    m = Model(
        None, None, None, None, vocab_file,
        num_units=1024, layers=4, dropout=0.2,
        batch_size=32, learning_rate=0.0001,
        output_dir=model_dir,
        restore_model=True, init_train=False, init_infer=True)
    if 'input' in request.GET and request.GET['input']:
        up = request.GET['input']
        output = m.infer(' '.join(up))
        output = ''.join(output.split(' '))
        return render(request, "1.htm", {"input": up, "up": up, "down": output})
    else:
        fileadd = "F:/001/data/对联.txt"
        f = open(fileadd, 'r', encoding='utf-8')
        data = f.read()
        n = data.count('\n')
        i = random.randint(1, n)
        up = linecache.getline(fileadd, i).replace(' ', '')
        output = m.infer(' '.join(up))
        output = ''.join(output.split(' '))
        f.close()
        return render(request, "1.htm", {"up": up, "down": output})

