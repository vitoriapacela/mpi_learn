import optparse

parser = optparse.OptionParser()
parser.add_option('--restart',action='store_true')
parser.add_option('--train',action='store_true')
parser.add_option('--test',action='store_true')
parser.add_option('--fresh',action='store_true')
parser.add_option('--fom', action='store_true')
parser.add_option('--tag',default='')
parser.add_option('--max',type='int', default = 0)
parser.add_option('--lr',type='float',default=0.0)
parser.add_option('--epochs', type='int', default=3)
parser.add_option('--cache',default=None)

(options,args) = parser.parse_args()

from mpi_learn.train.HGCALModel import GANModel
from mpi_learn.train.data import H5Data
import h5py
import setGPU
import json
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


gan_args = {
    #'tell': False,
    #'reversedorder' : False,
    #'heavycheck' : False,
    #'show_values' : False,
    #'gen_bn' : True,
    #'checkpoint' : False,
    #'onepass' : False,
    #'show_loss' : True,
    #'with_fixed_disc' : True ## could switch back to False and check
    }
    
gm = GANModel(**gan_args)

restart = options.restart
fresh = options.fresh
tag = (options.tag+'_') if options.tag else ''
lr = options.lr

if restart:
    tag+='reload_'
    print ("Reloading")
    if lr:
        gm.compile( prop = False, lr=lr)
        tag+='sgd%s_'%lr
    else:
        gm.compile( prop = True)
        tag+='rmsprop_'

    ## start from an exiting model
    gm.generator.load_weights('FullRunApr3/simple_generator.h5')
    gm.discriminator.load_weights('FullRunApr3/simple_discriminator.h5')
    gm.combined.load_weights('FullRunApr3/simple_combined.h5')
else:
    if lr:
        gm.compile(prop = False, lr=lr)
        tag+='sgd%s_'%lr
    else:
        gm.compile()
        tag+='rmsprop_'

    if not fresh:
        try:
            gm.generator.load_weights('simple_generator.h5')
            gm.discriminator.load_weights('simple_discriminator.h5')
        except:
            print ("fresh weights")
    else:
        tag+='fresh_'

print (tag,"is the option")

files = list(filter(None,open('train_3d_energy.list').read().split('\n')))
data = H5Data( batch_size = 100,
               cache = options.cache,
               preloading=0,
               features_name='X', labels_name='y')
data.set_file_names(files)
"""

if options.inmem:
    import os
    relocated = []
    os.system('mkdir /dev/shm/vlimant/')
    for fn in files:
        relocate = '/dev/shm/vlimant/'+fn.split('/')[-1]
        if not os.path.isfile( relocate ):
            print ("copying %s to %s"%( fn , relocate))
            if os.system('cp %s %s'%( fn ,relocate))==0:
                relocated.append( relocate )
    files = relocated
"""
history = {}
thistory = {}
fhistory = {}
etimes=[]
start = time.mktime(time.gmtime())


train_me = options.train
over_test= options.test
max_batch = options.max
ibatch=0
def dump():
    open('weights/simple_train_%s.json'%tag,'w').write(json.dumps(
        {
            'h':history,
            'th':thistory,
            'fh':fhistory,
            'et':etimes,
            } ))    

nepochs = options.epochs

def plotPred(sums, yc, i):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(sums, alpha=0.5, bins=50, label='generated')
    ax.hist(yc, alpha=0.5, bins=50, label='true')
    plt.legend()
    ax.set_xlabel('Energy (GeV)')
    ax.set_ylabel('n events')
    #plt.show()
    plt.savefig('figs/pred_%s.png'%i)

histories={}
for e in range(nepochs):
    history[e] = []
    thistory[e] = []
    fhistory[e] = []
    e_start = time.mktime(time.gmtime())
    for sub_X, sub_Y in data.generate_data():
        ibatch+=1
        #print (ibatch,ibatch>max_batch,max_batch)
        if over_test or not train_me:
            t_losses = gm.test_on_batch(sub_X,sub_Y)
            l = gm.get_logs( t_losses  ,val=True)
            gm.update_history( l , histories)
            t_losses = [list(map(float,l)) for l in t_losses]
            thistory[e].append( t_losses )
        if train_me:
            losses = gm.train_on_batch(sub_X,sub_Y)
            l = gm.get_logs( losses )
            gm.update_history( l , histories)                
            losses = [list(map(float,l)) for l in losses]
            history[e].append( losses )

            ## make plots
            yc = sub_Y[2]
            # number of events = batch size = len(sub_X) = 100
            noise = np.random.normal(0, 1, (len(sub_X), 200))
            sampled_energies = np.random.uniform(0.1, 5, (100, 1))
            generator_in = np.multiply(sampled_energies, noise)
            
            generated_images = gm.generator.predict(generator_in)
            inp_sums = np.sum(generated_images, axis=(1, 2, 3))
            plotPred(inp_sums, yc, e)

            # save weights after every epoch
            gm.generator.save_weights('weights_epoch/simple_generator_epoch_%s.h5'%e)
            gm.discriminator.save_weights('weights_epoch/simple_discriminator_epoch_%s.h5'%e)
            gm.combined.save_weights('weights_epoch/simple_combined_epoch_%s.h5'%e)

        if max_batch and ibatch>max_batch:
            break
            
        #if options.fom:
        #    fom = gm.figure_of_merit()
        #    print ("figure of merit",fom)

    if options.fom:
        fom = gm.figure_of_merit()
        print ("figure of merit",fom)
        fhistory[e].append( fom )
    gm.generator.save_weights('weights/simple_generator_%s.h5'%tag)
    gm.discriminator.save_weights('weights/simple_discriminator_%s.h5'%tag)
    gm.combined.save_weights('weights/simple_combined_%s.h5'%tag)
    dump()
    if max_batch and ibatch>max_batch:
        break
        
    e_stop = time.mktime(time.gmtime())
    print (e_stop - e_start,"[s] for epoch",e)
    etimes.append( e_stop - e_start)
dump()
