method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'gSTA'
hid_S = 64
hid_T = 256
N_T = 6
N_S = 2
# training
lr = 5e-3
drop_path = 0.2
batch_size = 8
sched = 'onecycle'

warmup_epoch = 5
epoch = 100
patience = 50
val_batch_size = 4