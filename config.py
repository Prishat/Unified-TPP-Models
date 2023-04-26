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

# Dataset Parameter Dictionaries

atm_params = {'seq_len': 10}

thp_data_params = {"data":"./mimic/fold1/",
                   "batch_size":4}

nhp_dataset_params = {'PathData':'./data/'}