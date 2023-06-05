# Model Parameter Dictionaries
rmtpp_params = {"model":"rmtpp",
                "seq_len":10,
                "emb_dim":10,
                "hid_dim":32,
                "mlp_dim":16,
                "dropout":0.1,
                "alpha":0.05,
                "batch_size":1024,
                "event_class":7,
                "verbose_step":350,
                "importance_weight":"store_true",
                "lr":1e-3,
                "epochs":30}

njsde_params = {'niters':100,
                'jump_type':'read',
                'batch_size':1,
                'nsave':10,
                'fold':0,
                'num_types':22,
                'restart':False,
                'evnt_align':False,
                'seed0':False,
                'debug':False}

thp_params = {'epoch':2,
              'batch_size':16,
              'd_model':64,
              'd_rnn':256,
              'd_inner_hid':128,
              'd_k':16,
              'd_v':16,
              'n_head':4,
              'n_layers':4,
              'dropout':0.1,
              'lr':1e-4,
              'smooth':0.1,
              'log':'log.txt',
              'num_types': 75, ## For Mimic dataset
              'device':'cpu'}

nhp_params = {'DimLSTM':16,
              'total_event_num':10,
              'PathData':'./data',
              'UseGPU':False,
              'Multiplier':1,
              'SizeBatch':50,
              'TrackPeriod':500,
              'MaxEpoch':10,
              'learning_rate':1e-3,
              'Seed':12345,
              'PathSave':'./savedmodel'}

# Dataset Parameter Dictionaries

atm_params = {'seq_len': 10,
              'batch_size':4}

thp_data_params = {"data":"./mimic/fold1/",
                   "batch_size":4}

nhp_dataset_params = {'PathData':'./data/'}

sahp_dataset_params = {'PathData':'./data/'}
