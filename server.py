from model import Model
import linecache
import random
import os
vocab_file = 'D:/AI/团队作业/001/data/dl-data/couplet/vocabs'
model_dir = 'data/dl-data/models/tf-lib/output_couplet/'
fileadd = 'data/dl-data/couplet/train/in.txt'

basedir = os.path.dirname(__file__)
m = Model(

    None, None, None, None, vocab_file,

    num_units=1024, layers=4, dropout=0.2,

    batch_size=32, learning_rate=0.0001,

    output_dir=model_dir,

    restore_model=True, init_train=False, init_infer=True)

f = open(fileadd,'r',encoding='utf-8')
data = f.read()
n=data.count('\n')
while(True):
    flag = input()
    if flag=='1':
        i = random.randint(1, n)
        user = linecache.getline(fileadd, i).replace(' ','')
    else:
        user = input()

    output = m.infer(' '.join(user))
    output = ''.join(output.split(' '))
    result = {user: output}
    print('上联：%s；下联：%s' % (user, output))

f.close()
