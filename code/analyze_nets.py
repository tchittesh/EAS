import json
from os.path import join, isfile

import tensorflow as tf

from expdir_monitor.expdir_monitor import ExpdirMonitor
from data_providers.utils import get_data_provider_by_name
from models.utils import get_model_by_name

net_pool_path = '../net_pool_base1/Convnet/C10+/Conv_C10+_rl_small'

id2val = json.load(open(join(net_pool_path, 'net.id2val')))
str2id = json.load(open(join(net_pool_path, 'net.str2id')))

id = list(id2val.keys())[0]
em = ExpdirMonitor(f'{net_pool_path}/#{id}')


pure = False
valid_size = -1

init = em.load_init()
print(init['layer_cascade']['layers'][0].keys())
run_config = em.load_run_config(print_info=(not pure), dataset='C10+')
run_config.renew_logs = False
if valid_size > 0:
    run_config.validation_size = valid_size

data_provider = get_data_provider_by_name(run_config.dataset, run_config.get_config())
net_config, model_name = em.load_net_config(init, print_info=(not pure))
model = get_model_by_name(model_name)(em.expdir, data_provider, run_config, net_config, pure=pure)
model._count_trainable_params()
start_epoch = 1

print('Testing...')
loss, accuracy = model.test(data_provider.test, batch_size=200)
print('mean cross_entropy: %f, mean accuracy: %f' % (loss, accuracy))
json.dump({'test_loss': '%s' % loss, 'test_acc': '%s' % accuracy}, open(em.output, 'w'))
